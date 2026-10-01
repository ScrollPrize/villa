"""Extract a recorded test image without changing training or image pixels."""
import hashlib
import io
import json

from PIL import Image
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from reviewed_fiber_evaluation import ROOT, HALFSIGMA_RAW
from lasagna.inference_provenance import atomic_write

STEP = 3980800
TAG = 'test_sample_3d/principal_slices'


def run():
    run_dir = HALFSIGMA_RAW.parent.parent
    output = ROOT/'halfsigma_current_raw_v2'/f'test_images_{STEP}'
    matches = []
    for event_file in sorted(run_dir.glob('events.out.tfevents.*')):
        for event in LegacyEventFileLoader(str(event_file)).Load():
            if event.step != STEP:
                continue
            for value in event.summary.value:
                if value.tag != TAG or not value.HasField('image'):
                    continue
                data = value.image.encoded_image_string
                with Image.open(io.BytesIO(data)) as image:
                    assert image.format == 'PNG'
                    assert image.size == (value.image.width,value.image.height)
                    image.verify()
                filename = f'test_principal_slices_samples_{STEP}_{len(matches):02d}.png'
                output.mkdir(parents=True,exist_ok=True)
                target = output/filename
                if target.exists() and target.read_bytes() != data:
                    raise ValueError(f'refusing to replace differing image: {target}')
                if not target.exists():
                    target.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
                matches.append(dict(path=str(target),event_file=str(event_file),tag=TAG,
                    step=STEP,wall_time=event.wall_time,width=value.image.width,
                    height=value.image.height,bytes=len(data),sha256=digest))
    if not matches:
        raise ValueError(f'no test image at {STEP}')
    atomic_write(output/'manifest.json',dict(training_run=str(run_dir),images=matches),durable=False)
    print(json.dumps(matches,indent=2),flush=True)


if __name__ == '__main__':
    run()
