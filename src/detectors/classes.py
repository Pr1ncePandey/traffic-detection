"""Grouping COCO classes into road-relevant categories.

Why this exists: the old code hardcoded `CLASS_NAMES = {2:"car", 3:"motorcycle",
5:"bus", 7:"truck"}` and filtered detection to those four ids, so nothing else
on the road was even seen. Now all 80 COCO classes are detected and each is
tagged with a GROUP, which lets an analyzer ask "is this a vehicle?" without
knowing a single class id. That is what keeps plate OCR cheap: the ANPR
analyzer runs on group == "vehicle" only, so a detected person or dog never
enters the OCR path.

HONEST LIMIT, and it matters for the "anything on road" requirement: COCO has
no class for pothole, debris, traffic cone, barrier, or fallen branch. Those
are not misclassified - they are simply invisible to this model. The OBSTACLE
group below is a pragmatic reuse of COCO classes that have no business being
on a carriageway (a suitcase, a chair, a bottle); it is a useful signal, not a
real obstacle detector. Naming arbitrary road objects needs an
open-vocabulary model (YOLO-World / YOLOE) behind the same Detector interface.
"""

VEHICLE = "vehicle"
PERSON = "person"
ANIMAL = "animal"
OBSTACLE = "obstacle"
INFRASTRUCTURE = "infrastructure"
OTHER = "other"

# Road vehicles only. airplane/boat are COCO classes but not road traffic, so
# they fall through to OTHER rather than inflating vehicle counts.
_VEHICLE = {"bicycle", "car", "motorcycle", "bus", "train", "truck"}

_PERSON = {"person"}

_ANIMAL = {"bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe"}

# Fixed roadside furniture: useful context, never a moving hazard.
_INFRASTRUCTURE = {"traffic light", "fire hydrant", "stop sign",
                   "parking meter", "bench"}

# Portable objects that signify debris or an obstruction when detected on a
# carriageway. Deliberately conservative - only things large or solid enough to
# matter to a driver.
_OBSTACLE = {"backpack", "handbag", "suitcase", "umbrella", "bottle",
             "chair", "couch", "potted plant", "bed", "dining table",
             "tv", "laptop", "microwave", "oven", "refrigerator", "sink",
             "toilet", "vase", "sports ball", "skateboard", "surfboard",
             "skis", "snowboard", "book", "teddy bear"}

_GROUPS = {}
for _name in _VEHICLE:
    _GROUPS[_name] = VEHICLE
for _name in _PERSON:
    _GROUPS[_name] = PERSON
for _name in _ANIMAL:
    _GROUPS[_name] = ANIMAL
for _name in _INFRASTRUCTURE:
    _GROUPS[_name] = INFRASTRUCTURE
for _name in _OBSTACLE:
    _GROUPS[_name] = OBSTACLE

# Groups that represent something actually on the road, as opposed to scenery.
ON_ROAD = (VEHICLE, PERSON, ANIMAL, OBSTACLE)


def group_of(cls_name: str) -> str:
    """Map a class name to its group. Unknown names -> "other", never an error."""
    return _GROUPS.get(str(cls_name).strip().lower(), OTHER)


def is_vehicle(cls_name: str) -> bool:
    """The gate for plate reading and vehicle counting."""
    return group_of(cls_name) == VEHICLE


def is_on_road(cls_name: str) -> bool:
    return group_of(cls_name) in ON_ROAD


def names_in_group(group: str) -> set:
    return {n for n, g in _GROUPS.items() if g == group}


def resolve_classes(spec, model_names: dict) -> list | None:
    """Turn the `model.classes` config value into ultralytics' `classes=` arg.

    Accepts:
      None / "all"        -> None (every class; what "anything on road" needs)
      "on_road"           -> the vehicle/person/animal/obstacle ids
      "vehicle" (a group) -> that group's ids
      [2, 3, 5, 7]        -> passed through, for backward compatibility
      ["car", "truck"]    -> names resolved to ids
    """
    if spec is None or (isinstance(spec, str) and spec.lower() in ("all", "", "none")):
        return None
    by_name = {str(v).lower(): int(k) for k, v in model_names.items()}
    if isinstance(spec, str):
        key = spec.lower()
        wanted = (set().union(*(names_in_group(g) for g in ON_ROAD))
                  if key == "on_road" else names_in_group(key))
        if not wanted:
            print(f"[classes] unknown class group {spec!r}, detecting all classes")
            return None
        return sorted(by_name[n] for n in wanted if n in by_name)
    ids = []
    for item in spec:
        if isinstance(item, int) or str(item).isdigit():
            ids.append(int(item))
        elif str(item).lower() in by_name:
            ids.append(by_name[str(item).lower()])
        else:
            print(f"[classes] unknown class {item!r}, ignoring")
    return sorted(set(ids)) or None
