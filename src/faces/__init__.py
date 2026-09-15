"""Face recognition: find enrolled people in camera footage (YuNet + AdaFace).

Merged from the standalone face-lab prototype (docs/face-recognition.md has
the full decision log and measurements). Library layout:

    errors.py     exceptions whose messages tell the operator what to fix
    detector.py   YuNet face detection (+ multi-scale search for photos)
    align.py      5-point alignment to 112x112, face quality signals
    embedder.py   AdaFace IR101 on onnxruntime -> 512-d unit vectors
    gallery.py    enrolled embeddings -> best person per face; photo helpers
    voting.py     per-track reads -> a name, or unknown with the reason
    people.py     the people database (name + photo paths + cached vectors)

The per-camera part - which faces belong to which tracked person, when to run
AdaFace, when a match is confirmed - is the `face` enricher in
src/attributes/enrichers/face.py. Enrol people with tools/people.py.

No tracker here: faces ride on the main pipeline's ByteTrack person tracks,
so a name lands on the same object id as the person's attributes and crop.
"""
