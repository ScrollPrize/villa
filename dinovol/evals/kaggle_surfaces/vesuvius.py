"""Import the upstream model components, optionally from a source checkout."""
from importlib import import_module
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from pathlib import Path
import sys


def components(source=None):
    if source is not None:
        # Explicit source mode exposes the models as a namespace package. This
        # avoids vesuvius.__init__ importing unrelated volume/network packages
        # (nrrd, etc.). All model/loss code still runs directly from upstream.
        package = Path(source).expanduser().resolve() / "vesuvius"
        if not (package / "models/build/decoder.py").is_file():
            raise FileNotFoundError(f"Expected a Vesuvius src directory: {source}")
        if "vesuvius" in sys.modules:
            if str(package) not in list(sys.modules["vesuvius"].__path__):
                raise RuntimeError("A different Vesuvius checkout is already imported")
        else:
            spec = ModuleSpec("vesuvius", loader=None, is_package=True)
            spec.submodule_search_locations = [str(package)]
            sys.modules["vesuvius"] = module_from_spec(spec)
    decoder = import_module("vesuvius.models.build.decoder").Decoder
    loss = import_module("vesuvius.models.training.loss.skeleton_recall").DC_SkelREC_and_CE_loss
    return decoder, loss


def medial_loss(source=None):
    _, loss = components(source)
    kwargs = dict(batch_dice=False, smooth=1e-5, do_bg=False, ddp=False)
    return loss(soft_dice_kwargs=kwargs.copy(), soft_skelrec_kwargs=kwargs.copy(),
                ce_kwargs={}, weight_ce=1, weight_dice=1, weight_srec=1, ignore_label=2)
