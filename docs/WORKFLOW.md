# VantaSight: কাজের ধাপ ও বর্তমান অবস্থা

এটি VantaSight তৈরির ধাপে ধাপে কাজের তালিকা। প্রতিটি ধাপ শুরুর আগে কী করতে যাচ্ছি তা জানাব। ধাপ শেষ হলে কী বদলেছে, কেন দরকার, কীভাবে ব্যবহার করতে হবে এবং কোনো সীমাবদ্ধতা আছে কি না—সহজ ভাষায় বুঝিয়ে দেব। কোনো কাজ যাচাই করা না গেলে সেটিও পরিষ্কার করে বলব।

## প্রকল্পের লক্ষ্য

প্রথমে Valorant-এর নিজের screenshot বা recorded gameplay video বিশ্লেষণ করে দৃশ্যমান তথ্য নিয়ে গবেষণা করা: character candidate শনাক্ত করা, দৃশ্যমান body-region অনুমান করা, পরপর frame-এ detection track করা, এবং crosshair reference থেকে target region-এর দূরত্ব মাপা। ভবিষ্যতে অন্য FPS বা online game-এ নেওয়ার উপযোগী করে input ও analytics-এর সাধারণ অংশগুলো game-specific অংশ থেকে আলাদা রাখা হবে। প্রকল্পটি game input পাঠাবে না, aim নড়াবে না, fire করবে না, game memory পড়বে না বা game process-এ প্রবেশ করবে না। প্রথম সংস্করণ শুধু offline footage বিশ্লেষণ করবে।

## এখন পর্যন্ত কী করা হয়েছে

- **M1 — Input foundation:** image loader, video metadata reader ও CLI লেখা এবং `vd-1.mp4` দিয়ে runtime-এ যাচাই করা হয়েছে।
- **M2 — Frame preview:** নির্দিষ্ট video frame পড়া ও preview image save করা হয়েছে। Frame index 3540 (29.5 সেকেন্ড) সফলভাবে বের হয়েছে: `data/previews/vd-1_frame_003540.png`। ছয়টি অতিরিক্ত সময়ে frame দেখে contact sheet বানানো হয়েছে: `data/previews/vd-1_contact-sheet.jpg`।
- **M3 — Baseline detection:** TorchVision COCO `person candidate` detector যোগ করে `ss-1`, `ss-2` এবং `vd-1`-এর frame 3000-এ runtime inference করা হয়েছে।
- প্রথম inference-এ `ss-1`-এর doorway character candidate confidence 0.286-এ ধরা পড়ে; HUD-তেও false positive ছিল। `ss-2`-এর in-world negative sample-এ 3টি HUD/weapon false positive আসে। `vd-1` frame 3000-এ character candidate confidence 0.963-এ ধরা পড়ে, তবে webcam/HUD/weapon-ও candidate হয়েছিল।
- এরপর প্রতিটি source-এর জন্য আলাদা exclusion rectangle `data/annotations/source_regions.json`-এ যোগ করা হয়েছে। Mask candidate বাদ দেয়, source frame crop বা resize করে না; detection box মূল resolution-এই ফেরত আসে।
- Mask-এর পর `ss-1`-এ doorway candidate (0.286) রয়ে গেছে। `ss-2`-এ একটি arm/weapon false positive (0.256) এখনো আছে। `vd-1` frame 3000-এ gameplay candidate (0.963) রয়ে গেছে, সঙ্গে first-person arm/weapon-এর 2টি false positive আছে।
- Mask কিছু HUD ও webcam false positive কমিয়েছে, কিন্তু baseline model এখনো Valorant enemy চিনতে পারে না। এগুলো হাতে দেখা sample; accuracy metric নয়।
- `vd-1`-এর আরও পাঁচটি frame—5s, 25s, 35s, 45s ও 55s—GPU-তে বিশ্লেষণ করা হয়েছে। 25s ও 45s-এ দূরের opponent candidate ধরা পড়েছে; 5s, 35s ও 55s-এ দৃশ্যমান opponent না থাকলেও হাতে/অস্ত্রে, overlay বা artwork-এ ভুল candidate এসেছে। 35s-এর stream inset আগের mask-এর জায়গায় ছিল না—ভিডিওতে overlay-এর অবস্থান বদলাতে পারে।
- পাঁচটি frame-এর হাতে দেখা ফল `data/annotations/vd-1_frame_review.csv`-এ রাখা হয়েছে: মোট 2টি সঠিক candidate এবং 9টি false candidate। এটি পাঁচটি নির্বাচিত frame-এর review মাত্র; পুরো video-র accuracy নয়।
- Annotated frame-গুলো `data/previews/vd-1_frame_000600_detections.png` থেকে `vd-1_frame_006600_detections.png` পর্যন্ত আছে; একসঙ্গে দেখার sheet `data/previews/vd-1_detection_review_sheet.jpg`। এগুলো local ignored output।
- একই পাঁচটি frame-এ confidence threshold `0.25`, `0.50` ও `0.75` তুলনা করা হয়েছে। Candidate count যথাক্রমে 11, 8 ও 5; হাতে চিহ্নিত সঠিক candidate 2, 1 ও 1; false candidate 9, 7 ও 4। `0.50` এবং `0.75`-এ 45s-এর ছোট opponent (confidence `0.322`) বাদ পড়ে, কিন্তু কিছু ভুল hand/weapon candidate উচ্চ confidence-এ টিকে যায়।
- Threshold-ভিত্তিক candidate ও box-এর ফল `data/annotations/vd-1_threshold_comparison.csv`-এ আছে; পাশাপাশি তুলনার preview `data/previews/vd-1_threshold_comparison.jpg`। Threshold বাড়ালে false positive কিছু কমে, তবে ছোট opponent miss হয়—তাই default `0.25` আপাতত বদলানো হয়নি।
- **M5 — Body-region experiment:** TorchVision Keypoint R-CNN-এর COCO_V1 দিয়ে `ss-1`, `vd-1` frame 3000 (25s) ও frame 5400 (45s)-এ GPU inference হয়েছে। `ss-1`-এ 1টি character candidate, 25s-এ 2টি candidate (gameplay opponent ও first-person object), 45s-এ 0 candidate এসেছে। 45s-এ detector opponent পেয়েছিল, pose model পায়নি—দূরের ছোট target-এর জন্য এটি স্পষ্ট miss।
- Pose preview-তে magenta dot হলো model keypoint, রঙিন cross হলো keypoint থেকে গাণিতিকভাবে নেওয়া coarse region center। এগুলো অঙ্গ শনাক্তকরণের validated label নয়। Preview: `ss-1_body_regions.png`, `vd-1_frame_003000_body_regions.png`, `vd-1_frame_005400_body_regions.png` (`data/previews/`-এ)।
- TorchVision-এর keypoint heatmap score raw response, probability নয়। Code ও display-তে তাই `heatmap score` বলা হয়; এগুলোকে calibrated confidence হিসেবে দেখানো হয় না। Model COCO photo-তে প্রশিক্ষিত, Valorant-এর জন্য নয় এবং region accuracy এখনো মাপা হয়নি।
- **M6 — Temporary tracking:** `src/vantasight/tracking/iou_tracker.py` IoU overlap দিয়ে box-গুলোর short-term ID দেয়; matched box center-এর displacement ও video timestamp থেকে pixel/second velocity হিসাব করে। Missing detection পেলে সীমিত update পর্যন্ত অপেক্ষা করে; পরে track শেষ করে।
- `vd-1`-এর 25.0–25.5s segment-এ 10-frame interval-এ 7টি frame চালানো হয়েছে: 28টি person candidate, 15টি temporary ID তৈরি হয়েছে। দৃশ্যমান opponent ID 2-এ শুধু 25.000s ও 25.083s sample-এ ছিল; পরের frame-গুলোতে detector তাকে miss করে। Webcam/arm/weapon false candidate-ও আলাদা ID পেয়েছে। তাই tracking output-কে player identity বা valid target track বলা যাবে না।
- ওই smoke run-এর measured analysis throughput 3.423 sampled frame/s; detector latency mean 178.2ms, p95 487.9ms। মাত্র 7 sample ও একটি ছোট segment—এই speed-কে পুরো video বা সব hardware-এর performance হিসেবে ধরা যাবে না। CSV `reports/vd-1_tracking_10frame.csv`; preview `data/previews/vd-1_frame_003010_tracks.png`।
- **M7 — Alignment geometry:** crosshair reference থেকে available estimated region center পর্যন্ত pixel distance ও frame diagonal দিয়ে normalized distance হিসাব করা হয়েছে। `ss-1`-এ frame center `(958.5, 536.5)` ধরে head-estimate distance `24.75px` / normalized `0.011265`; এটি pose model-এর অনুমানের ওপর নির্ভরশীল, target-hit বা aim accuracy নয়।
- OpenCV `5.1.0.95` ও project package virtual environment-এ install হয়েছে।
- `vd-1.mp4`-এর metadata: 59 সেকেন্ড, 2560×1440, 120 FPS, 7,080 frame, ফাইল প্রায় 203 MB।
- 5, 15, 25, 35, 45 ও 55 সেকেন্ডের frame sample দেখা হয়েছে; এটি পুরো video review নয়। 25 ও 45 সেকেন্ডের sample-এ opponent দেখা যায়, তবে দূরে ও ছোট। সব frame-এ enemy থাকবে এমন নয়—enemy কিছু সময় অনুপস্থিত থেকে পরে দেখা দিতে পারে, সেটি স্বাভাবিক।
- তোমার ব্যাখ্যা অনুযায়ী stream-এর webcam inset-এর মানুষ game-এর enemy নয় এবং ওই inset enemy-কে ঢাকে না। তাই webcam-কে enemy detection হিসেবে গণনা করব না, তবে পুরো game frame বাদও দেব না।
- Original frame 2560×1440 resolution-এ রাখা হয়েছে। Contact sheet-এর thumbnail-গুলো শুধু একসঙ্গে দেখার সুবিধার জন্য ছোট করা; ওই thumbnail দিয়ে enemy size বিচার করা যাবে না।
- `data/input/screenshots/` screenshot রাখার জন্য তৈরি।
- `data/input/videos/` recorded video রাখার জন্য তৈরি।
- `data/previews/` তৈরি হওয়া preview রাখার জন্য তৈরি।
- দেওয়া দুটি screenshot `ss-1.png` ও `ss-2.png` নামে input folder-এ রাখা হয়েছে; resolution যথাক্রমে 1917×1073 এবং 1917×1070। Git এগুলো ignore করছে।
- প্রাথমিক visual label: `ss-1`-এ gameplay doorway-তে enemy character দৃশ্যমান; `ss-2`-এ main sightline-এ সাধারণ gameplay character দৃশ্যমান নয়। `ss-2`-এর লাল replay silhouette-গুলোকে আলাদা overlay হিসেবে ধরব, normal gameplay player detection-এর ground truth হিসেবে নয়।
- Video ও screenshot-এর ID আলাদা ধারায় চলবে: `vd-1`, `vd-2`, ... এবং `ss-1`, `ss-2`, ...। File extension আগের মতো থাকবে, যেমন `vd-1.mp4` বা `ss-1.png`।
- `README.md`-তে setup ও ব্যবহারের নির্দেশনা আছে। `.gitignore` input media ও generated preview Git থেকে বাদ রাখে।
- Detector-এর weight source, license note ও known limitation `docs/MODEL_CARDS.md`-এ লেখা আছে।
- Screenshot-এর user-provided positive/negative label `data/annotations/screenshot_presence.csv`-এ রাখা হয়েছে; annotation data Git ignore করা।

Enemy box হাতে label করার interactive local tool `tools/annotate_video_frames.py` এবং label থেকে precision/recall হিসাবের tool `tools/evaluate_detector.py` যোগ করা হয়েছে। Frame preview ছোট করে window-তে দেখালেও box original video pixel coordinate-এ save হবে। Annotation GUI-নির্ভর; এই সেশনে key/mouse দিয়ে পাঁচটি frame label করা হয়নি, তাই ground-truth box ও metric এখনো তৈরি হয়নি। Body-region estimation-এর experimental model ও preview আছে, কিন্তু সেটিও labeled ground truth দিয়ে যাচাই হয়নি। Tracker, alignment calculation ও dashboard এখনো তৈরি হয়নি।

## কাজের roadmap

| ধাপ | কাজ | ধাপ শেষে যা পাওয়া যাবে | অবস্থা |
|---|---|---|---|
| 1. Input foundation | Python package, screenshot/video reader, metadata | Image ও video load করার ভিত্তি | `vd-1.mp4` দিয়ে runtime-এ যাচাই হয়েছে |
| 2. Frame preview | Video থেকে নির্দিষ্ট frame বের করা ও preview image save করা | বাস্তব footage-এর নির্দিষ্ট frame দেখা | Frame 3540 সফলভাবে বের ও দেখা হয়েছে |
| 3. প্রাথমিক detection | COCO pretrained person-candidate model; confidence ও bounding box দেখানো | Model কোন জিনিস শনাক্ত করছে তা পর্যালোচনা | প্রথম inference ও per-source mask হয়েছে; false positive বাকি |
| 4. Dataset review | বাস্তবসম্মত clip-এ false positive ও missed detection হাতে পর্যালোচনা | Detector উপযোগী কি না তার প্রাথমিক প্রমাণ | label ও metric tool তৈরি; পাঁচটি frame-এর ground-truth label বাকি |
| 5. Body-region পরীক্ষা | Pose/region model চেষ্টা করা ও অনিশ্চয়তা দেখানো | Score ও uncertainty-সহ region estimate | COCO Keypoint R-CNN-এর প্রথম inference হয়েছে; 45s-এর ছোট opponent miss, accuracy যাচাই বাকি |
| 6. Tracking | একাধিক frame-এর detection মিলিয়ে temporary ID দেওয়া | Track কতটা ধারাবাহিক তা দেখা | IoU tracker, CSV/JSON export আছে; ছোট segment-এ চালানো হয়েছে, detector miss-এ track ভেঙেছে |
| 7. Alignment metrics | Crosshair reference সেট করা এবং pixel/normalized distance হিসাব | জ্যামিতিক alignment measurement | math module, CLI ও unit tests আছে; `ss-1`-এ runtime check হয়েছে |
| 8. Export ও dashboard | CSV/JSON এবং আলাদা analysis preview/dashboard | ব্যবহারযোগ্য research prototype | CSV/JSON export ও local report dashboard যোগ হয়েছে; Windows-এ UI visual review বাকি |
| 9. Evaluation ও documentation | Annotated sample-এ metric মাপা, limitation লেখা | পুনরায় যাচাইযোগ্য ফল ও setup guide | Maximum one-to-one detector matching ও metric module আছে; representative frame label ও pose-region evaluation বাকি |

প্রতিটি পরের ধাপ আগের ধাপের ফলের ওপর ভিত্তি করে হবে। কোনো model বাস্তব footage-এ দুর্বল ফল দিলে সেটা স্পষ্টভাবে জানানো হবে; প্রয়োজন হলে model, data বা পদ্ধতি বদলে আবার মাপা হবে।

## M1 setup করার নিয়ম

PowerShell-এ repository folder-এ গিয়ে চালাও:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Screenshot inspect করতে:

```powershell
python -m vantasight "data/input/screenshots/screenshot.png"
```

`vd-1.mp4` নামে video এবং `ss-1.png` নামে screenshot রাখার উদাহরণ। পরের video হবে `vd-2.mp4`, পরের screenshot `ss-2.png`; video ও screenshot-এর serial আলাদা করে বাড়বে। দেওয়া ID file name-এ অপরিবর্তিত রাখা হবে।

Video metadata দেখতে:

```powershell
python -m vantasight "data/input/videos/clip.mp4"
```

## M2: video থেকে preview frame বের করা

Frame index শূন্য থেকে শুরু হয়: `0` হলো প্রথম frame, `120` হলো 121তম frame। Preview default-এ `data/previews/`-এ save হবে:

```powershell
python -m vantasight "data/input/videos/vd-1.mp4" --frame-index 120
```

নিজের output path দিতে চাইলে:

```powershell
python -m vantasight "data/input/videos/vd-1.mp4" --frame-index 120 --output "data/previews/vd-1_frame_000120.png"
```

Python code থেকে একটি নির্দিষ্ট frame পড়ার উদাহরণ:

```python
from vantasight.input import VideoReader

with VideoReader("data/input/videos/clip.mp4") as reader:
    frame = reader.read_frame(120)
    print(frame.index, frame.timestamp_seconds, frame.pixels.shape)
```

## M3: person-candidate detection

Detector class হিসেবে `person candidate` ফেরত দেয়; enemy/team পরিচয় দাবি করে না। Screenshot-এ চালাতে:

```powershell
python -m vantasight "data/input/screenshots/ss-1.png" --detect-persons
python -m vantasight "data/input/screenshots/ss-2.png" --detect-persons
```

Video frame 3000-এ চালাতে:

```powershell
python -m vantasight "data/input/videos/vd-1.mp4" --frame-index 3000 --detect-persons --max-side 2560 --exclude-region 0,620,530,960
```

Model image-কে ভিতরে resize করতে পারে, কিন্তু bounding box original source resolution-এ ফেরত দেয়। Per-file HUD/webcam exclusion `data/annotations/source_regions.json`-এ রাখা হয়; `--exclude-region x1,y1,x2,y2` দিয়ে অতিরিক্ত region যোগ করা যায়। এগুলো full-resolution frame crop করে না—চিহ্নিত region-এর মধ্যে center থাকা detection candidate বাদ দেয়। Replay silhouette ও game HUD-কে enemy বলে ধরে নেওয়া হবে না।

বর্তমান baseline COCO dataset-এর ordinary-person example-এ প্রশিক্ষিত; Valorant footage-এর জন্য তৈরি নয়। তাই result কেবল candidate—miss ও false positive হাতে পর্যালোচনা করতে হবে। Weight ও license detail: [MODEL_CARDS.md](MODEL_CARDS.md)।

## M4: opponent box হাতে label করা

Ground-truth box তৈরি করতে local desktop-এ চালাও:

```powershell
python tools/annotate_video_frames.py "data/input/videos/vd-1.mp4" --frames 600,3000,4200,5400,6600
```

Window-তে mouse টেনে প্রতিটি দৃশ্যমান in-world opponent-কে box করো। `y` = opponent দেখা যাচ্ছে, `n` = opponent দেখা যাচ্ছে না, `u` = নিশ্চিত নও, `Enter` = save করে পরের frame, `z` = শেষ box undo, `c` = frame-এর box মুছে ফেলা, `q` = বন্ধ করা। Box মূল video resolution-এর pixel coordinate-এ save হয়; preview ছোট করা হলেও input frame বদলায় না। Streamer inset, teammate, নিজের first-person হাত/অস্ত্র, HUD বা replay silhouette-কে opponent box করবে না।

প্রতিটি frame নিশ্চিত করে save করলে annotation `data/annotations/manual_enemy_boxes.jsonl`-এ append হবে; আগে save করা frame আবার skip হবে। Output file Git-ignored এবং source video-ও পরিবর্তিত হয় না।

Label করার পর metric চালাও:

```powershell
python tools/evaluate_detector.py "data/input/videos/vd-1.mp4" --confidence-threshold 0.25 --iou-threshold 0.50
```

Evaluator এক frame-এর detector box ও hand-labeled opponent box-কে IoU threshold দিয়ে one-to-one মেলায়। এখন matching algorithm threshold পেরোনো সব সম্ভাব্য pair থেকে maximum number of matches খোঁজে; শুধু সবচেয়ে বড় IoU pair আগে নেওয়ার কারণে বৈধ match বাদ পড়ে না। Uncertain frame বাদ পড়ে; CSV `reports/detector_evaluation.csv`-এ যায়। মোট TP/FP/FN থেকে precision, recall ও F1 বের হয়। Label file তৈরি না হওয়া পর্যন্ত metric বানানো হবে না, আর threshold comparison-কে ground-truth evaluation বলা যাবে না।

## M5: experimental body-region estimate

একটি screenshot বা video frame-এ চালাও:

```powershell
python -m vantasight "data/input/videos/vd-1.mp4" --frame-index 3000 --estimate-body-regions --max-side 1920
```

Output-এ COCO keypoint-গুলো এবং সেগুলো থেকে নেওয়া head/chest/lower-torso/legs region center দেখা যায়। `--keypoint-score-threshold` দিয়ে raw heatmap response-এর cutoff বদলানো যায়; raw response probability বা calibrated confidence নয়। Missing point/region দেখানো হবে না—অনুমান করে পূরণ করা হবে না। Model card: [MODEL_CARDS.md](MODEL_CARDS.md)।

প্রথম run-এ `ss-1` ও `vd-1` frame 3000-এ body point পাওয়া গেছে; frame 5400-এ কোনো person candidate আসেনি, যদিও baseline detector সেখানে দূরের opponent candidate পেয়েছিল। তাই বর্তমান keypoint model ছোট/distant opponent-এর জন্য যথেষ্ট নয়। Preview-গুলো qualitative inspection মাত্র, region accuracy নয়।

## M6: sampled video candidate tracking

সংক্ষিপ্ত segment-এ চালানোর উদাহরণ:

```powershell
python tools/track_video.py "data/input/videos/vd-1.mp4" --start-frame 3000 --stop-frame 3061 --frame-step 10 --preview-frame 3010 --output reports/vd-1_tracking.csv
```

`--frame-step 10` মানে প্রতি 10 source frame-এ inference। CSV-তে bounding box, temporary track ID, confidence এবং image-plane velocity (pixel/second) থাকে। Source FPS আলাদা করে দেখায়; inference throughput ও detector latency-ও terminal-এ মাপে। `--stop-frame` শেষ সীমা, সেটি অন্তর্ভুক্ত নয়।

একই run থেকে `reports/tracks.json`-এ structured report-ও তৈরি হয়। এতে video metadata, ব্যবহৃত threshold ও sampling settings, per-frame candidate detection/box/temporary ID/velocity এবং run summary থাকে। `--json-output` দিয়ে অন্য path বেছে নেওয়া যায়। JSON schema-র `schema_version` ভবিষ্যৎ পরিবর্তন চিহ্নিত করবে। CSV ও JSON-ই ignored `reports/` folder-এ থাকে।

### M8: আলাদা offline dashboard

Saved JSON report ও তার source video দেখতে:

```powershell
python tools/open_dashboard.py reports/tracks.json
```

Dashboard sampled frame-এর তালিকা দেয়; frame নির্বাচন করলে local video থেকে ওই frame পড়ে JSON report-এর candidate box/temporary ID আঁকে। পাশে timestamp, detector score, box coordinate ও image-plane velocity দেখা যায়। অন্য folder-এর video হলে `--video path\to\clip.mp4` দাও। Dashboard live screen/game capture করে না। এই environment থেকে Windows window-তে click করে visual review করা যায়নি; স্থানীয় desktop-এ চালিয়ে UI layout দেখে নিতে হবে।

25.0–25.5s sample-এ visible opponent মাত্র 2টি sampled frame পর্যন্ত একই ID পেয়েছে, এরপর detector তাকে miss করেছে। First-person/overlay false candidate-ও track হয়। IoU tracker দ্রুত movement, box-shape পরিবর্তন ও missed detection-এ ID fragment করতে পারে; ফলে এটি baseline association experiment, validated tracking নয়।

## M7: crosshair-to-region alignment

Frame center-কে default reference ধরে estimate মাপতে:

```powershell
python -m vantasight "data/input/screenshots/ss-1.png" --estimate-body-regions --measure-alignment
```

নিজের crosshair reference original source pixel-এ দিতে পারো, উদাহরণ: `--crosshair 959,536`। Pixel distance হলো region center ও reference-এর Euclidean distance; normalized distance = pixel distance / `sqrt(width² + height²)`। Preview-তে red cross reference দেখায়।

`ss-1`-এ default frame-center দিয়ে head estimate-এর distance 24.75px, normalized 0.011265 পাওয়া গেছে। Region centerটি COCO Keypoint R-CNN-এর অনুমান, তাই এই measurement-ও অনুমাননির্ভর। এটি geometric offset—enemy identity, shot hit, accuracy percentage বা aim automation নয়।

## ছবি ও ভিডিও GitHub-এ যাবে কি?

`data/input/screenshots/` এবং `data/input/videos/`-এর media আপাতত Git ignore করা আছে, তাই স্বাভাবিক commit/push-এ যাবে না। তুমি gameplay video, screenshot ও preview GitHub-এ upload করার অনুমতি দিয়েছ। বড় video-র জন্য Git LFS বা অন্য উপযুক্ত storage পদ্ধতি ঠিক করে পরে নির্বাচিত sample upload করব; generated preview-ও প্রয়োজনমতো বেছে নেওয়া হবে।

Folder-গুলো খালি থাকলেও Git-এ দেখানোর জন্য ছোট `.gitkeep` placeholder file রাখা আছে। এগুলোতে ছবি বা ভিডিও থাকে না।

## আমরা যেভাবে কাজ করব

1. কাজ শুরুর আগে লক্ষ্য এবং কোন file বা অংশে পরিবর্তন হতে পারে তা জানাব।
2. প্রয়োজনীয় code বা documentation update করব এবং dependency বা permission সমস্যা থাকলে জানাব।
3. ধাপ শেষে কী বদলেছে, কীভাবে ব্যবহার করতে হবে, কী যাচাই হয়েছে এবং কী সীমাবদ্ধতা আছে—তা বুঝিয়ে বলব।
4. বাস্তব footage লাগলে তোমার দেওয়া screenshot বা ছোট recorded clip দিয়ে ফল পর্যালোচনা করব।

## পরের কাজের জন্য যা দরকার

M1–M3-এর baseline, M5 body-region experiment, M6 sampled tracking, M7 alignment calculation এবং M8 CSV/JSON export ও offline dashboard আছে। M9 detector evaluator-এ one-to-one maximum matching যোগ হয়েছে; এখন representative gameplay frame label ছাড়া বাস্তব metric বাকি। পরের কাজ local annotation window-তে frame 600, 3000, 4200, 5400 ও 6600-এর opponent box label করে evaluator চালানো। Pose-region estimate যাচাইয়ের label format ও metric-ও যোগ করতে হবে। Label ছাড়া detector, pose বা tracking accuracy দাবি করা যাবে না।

## ভাষার নিয়ম

এই workflow নথিটি Bangla-তে থাকবে, যাতে ধাপগুলো অনুসরণ করা সহজ হয়। Code, variable, command ও মূল project documentation আপাতত English-এ থাকবে। Project শেষ হলে প্রয়োজন অনুযায়ী পুরো project-কে consistent international English-এ নেওয়া হবে।
