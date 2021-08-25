# Music-recognition

## Installation
```bash
pip install -e .
```
To get the `background data` 
```
pip install git-lfs 
git lfs install
git lfs fetch --all
git lfs pull
```

## Developer note
Background noises can be split into segments
```bash
for i in *.ogg; do ffmpeg -i "$i" -f segment -segment_time 30 -c copy "../background_segments/${i%.*}_%03d.ogg"; done
```
