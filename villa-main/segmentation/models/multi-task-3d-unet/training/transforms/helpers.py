import random

import volumentations as V

def apply_volumentations_to_all(
    data_dict: dict,
    transforms_list,
    p=1.0
):
    """
    Dynamically creates a volumentations pipeline that treats
    all keys in data_dict as 'image', EXCEPT 'normals'.

    One of these keys will be designated as the primary 'image',
    and the rest become 'additional_targets' so they get the
    same transforms.

    Args:
        data_dict (dict): e.g. {"image": np.array(...), "sheet": np.array(...), ...}
        transforms_list (list): a list of volumentations transforms
        p (float): Probability for the entire Compose.

    Returns:
        dict: the augmented data_dict
    """
    # if data_dict is empty or has only normals, just return
    if len(data_dict) == 0 or all(k == "normals" for k in data_dict):
        return data_dict

    # pick any key that is NOT "normals" to be the main 'image' key
    # e.g. if your data_dict is {'image': ..., 'sheet': ..., 'normals': ...}
    # we might pick 'image' as the primary
    # or if your data_dict has arbitrary keys, pick the first one you find
    main_key = None
    for k in data_dict:
        if k != "normals":
            main_key = k
            break

    # if everything was "normals" (which should be unusual), just return
    if main_key is None:
        return data_dict

    # Build additional_targets for all the other keys except 'normals' and the chosen main_key
    additional_targets = {}
    for k in data_dict:
        if k not in ["normals", main_key]:
            # treat them as "image" (so they get the same geometric transforms)
            additional_targets[k] = "image"

    # Create the pipeline with dynamic additional_targets
    pipeline = V.Compose(
        transforms_list,
        p=p,
        additional_targets=additional_targets
    )

    # Prepare the kwargs for pipeline(**kwargs)
    # main_key → "image", the others → the same key name
    vol_input = {}
    vol_input["image"] = data_dict[main_key]
    for k in data_dict:
        if k not in ["normals", main_key]:
            vol_input[k] = data_dict[k]

    # Apply the pipeline
    vol_output = pipeline(**vol_input)

    # Now, vol_output["image"] is the transformed version of data_dict[main_key].
    # vol_output[k] is the transformed version of that key for any other k.
    # Put everything back into data_dict
    data_dict[main_key] = vol_output["image"]
    for k in data_dict:
        if k not in ["normals", main_key] and k in vol_output:
            data_dict[k] = vol_output[k]

    # Return the augmented dictionary
    return data_dict
