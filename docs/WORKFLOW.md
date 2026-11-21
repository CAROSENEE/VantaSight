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

Enemy box label করার interactive local tool `tools/annotate_video_frames.py` এবং label থেকে precision/recall হিসাবের tool `tools/evaluate_detector.py` আছে। Frame preview ছোট করে window-তে দেখালেও box original video pixel coordinate-এ save হয়। `vd-1`-এর 12টি frame visually review করে local `data/annotations/vd-1_provisional_review.jsonl` বানানো হয়েছে; এতে প্রতিটি row-তে `assistant_visual_provisional_user_review_needed` note আছে, তাই এগুলো user-verified ground truth নয়। 15s, 30s, 40s ও 55s frame uncertain রাখা হয়েছে; বাকিগুলোতে 10s, 20s, 25s ও 45s-এ opponent visible এবং 0s, 5s, 35s ও 50s-এ sightline-এ opponent visible নয় বলে provisional label হয়েছে। এই 8টি নিশ্চিত frame-এ detector `TP=3, FP=24, FN=1`, precision `0.111`, recall `0.750`, F1 `0.194` দিয়েছে (`confidence 0.25`, `IoU 0.50`)। 20s-এর visible target-এ oversized candidate box-এর কারণে IoU threshold পাস হয়নি। Sample ছোট ও label provisional—metric-টি কেবল বর্তমান COCO baseline-এর দুর্বলতা/পরের data-collection প্রয়োজন বোঝায়। Body-region estimate এখনো কোনো manual point label দিয়ে যাচাই হয়নি।

## কাজের roadmap

| ধাপ | কাজ | ধাপ শেষে যা পাওয়া যাবে | অবস্থা |
|---|---|---|---|
| 1. Input foundation | Python package, screenshot/video reader, metadata | Image ও video load করার ভিত্তি | `vd-1.mp4` দিয়ে runtime-এ যাচাই হয়েছে |
| 2. Frame preview | Video থেকে নির্দিষ্ট frame বের করা ও preview image save করা | বাস্তব footage-এর নির্দিষ্ট frame দেখা | Frame 3540 সফলভাবে বের ও দেখা হয়েছে |
| 3. প্রাথমিক detection | COCO pretrained person-candidate model; confidence ও bounding box দেখানো | Model কোন জিনিস শনাক্ত করছে তা পর্যালোচনা | প্রথম inference ও per-source mask হয়েছে; false positive বাকি |
| 4. Dataset review | বাস্তবসম্মত clip-এ false positive ও missed detection হাতে পর্যালোচনা | Detector উপযোগী কি না তার প্রাথমিক প্রমাণ | চার frame-এর provisional visual metric আছে; user-verified labels ও বড় sample বাকি |
| 5. Body-region পরীক্ষা | Pose/region model চেষ্টা করা ও অনিশ্চয়তা দেখানো | Score ও uncertainty-সহ region estimate | COCO Keypoint R-CNN-এর প্রথম inference হয়েছে; 45s-এর ছোট opponent miss, accuracy যাচাই বাকি |
| 6. Tracking | একাধিক frame-এর detection মিলিয়ে temporary ID দেওয়া | Track কতটা ধারাবাহিক তা দেখা | IoU tracker, CSV/JSON export আছে; ছোট segment-এ চালানো হয়েছে, detector miss-এ track ভেঙেছে |
| 7. Alignment metrics | Crosshair reference সেট করা এবং pixel/normalized distance হিসাব | জ্যামিতিক alignment measurement | math module, CLI ও unit tests আছে; `ss-1`-এ runtime check হয়েছে |
| 8. Export ও dashboard | CSV/JSON এবং আলাদা analysis preview/dashboard | ব্যবহারযোগ্য research prototype | CSV/JSON export, body-region/alignment overlay ও manual-evaluation summary-সহ tracking dashboard যোগ হয়েছে; Windows-এ UI visual review বাকি |
| 9. Evaluation ও documentation | Annotated sample-এ metric মাপা, limitation লেখা | পুনরায় যাচাইযোগ্য ফল ও setup guide | Evaluator ও সীমাবদ্ধতা আছে; provisional metric হয়েছে, user-verified representative labels ও pose-region metric বাকি |

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

Window-তে `r` চেপে box mode-এ mouse drag করে প্রতিটি দৃশ্যমান in-world opponent-কে box করো। `y` = opponent দেখা যাচ্ছে, `n` = opponent দেখা যাচ্ছে না, `u` = নিশ্চিত নও, `Enter` = save করে পরের frame, `z` = শেষ box undo, `c` = সব box/point মুছে ফেলা, `q` = বন্ধ করা। Box মূল video resolution-এর pixel coordinate-এ save হয়; preview ছোট করা হলেও input frame বদলায় না। Streamer inset, teammate, নিজের first-person হাত/অস্ত্র, HUD বা replay silhouette-কে opponent box করবে না।

একই frame-এ pose-region যাচাইয়ের জন্য box আঁকার পর box-এর ভেতরে point চিহ্নিত করা ঐচ্ছিক: `1` = head, `2` = chest, `3` = lower torso, `4` = legs; এরপর selected opponent box-এর ভেতরের ওই region center-এ একবার click করো। `[` ও `]` দিয়ে একাধিক box-এর মধ্যে selected box বদলাও। শুধু detector evaluation করতে চাইলে কোনো region point না দিয়েও frame save করো।

প্রতিটি frame নিশ্চিত করে save করলে annotation `data/annotations/manual_enemy_boxes.jsonl`-এ append হবে; আগে save করা frame আবার skip হবে। Output file Git-ignored এবং source video-ও পরিবর্তিত হয় না।

Label করার পর metric চালাও:

```powershell
python tools/evaluate_detector.py "data/input/videos/vd-1.mp4" --confidence-threshold 0.25 --iou-threshold 0.50
```

Evaluator এক frame-এর detector box ও hand-labeled opponent box-কে IoU threshold দিয়ে one-to-one মেলায়। এখন matching algorithm threshold পেরোনো সব সম্ভাব্য pair থেকে maximum number of matches খোঁজে; শুধু সবচেয়ে বড় IoU pair আগে নেওয়ার কারণে বৈধ match বাদ পড়ে না। Uncertain frame বাদ পড়ে; CSV `reports/detector_evaluation.csv`-এ যায়। মোট TP/FP/FN থেকে precision, recall ও F1 বের হয়। Label file তৈরি না হওয়া পর্যন্ত metric বানানো হবে না, আর threshold comparison-কে ground-truth evaluation বলা যাবে না।

Pose-region point চিহ্নিত করে থাকলে model center-এর সঙ্গে তুলনা চালাও:

```powershell
python tools/evaluate_pose_regions.py "data/input/videos/vd-1.mp4"
```

এটি প্রতিটি region-এর pixel/normalized localization error এবং estimate coverage দেখায়; CSV `reports/pose_region_evaluation.csv`-এ যায়। Model region না পেলে error বানানো হয় না—coverage কমে। কয়েকটি হাতে label করা frame দিয়ে Valorant জুড়ে model-এর সাধারণ accuracy প্রমাণ হয় না।

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

একই run থেকে `reports/tracks.json`-এ structured report-ও তৈরি হয়। এতে video metadata, ব্যবহৃত threshold ও sampling settings, per-frame candidate detection/box/temporary ID/velocity এবং run summary থাকে। `--json-output` দিয়ে অন্য path বেছে নেওয়া যায়। JSON schema-র `schema_version` ভবিষ্যৎ পরিবর্তন চিহ্নিত করবে। CSV ও JSON-ই ignored `reports/` folder-এ থাকে। `--estimate-body-regions` দিলে sampled frame-এ pose model চালিয়ে candidate box-এর সঙ্গে one-to-one IoU association হয়; `--pose-match-iou-threshold` দিয়ে minimum overlap বদলানো যায়। Pose model-এর person detection threshold `--pose-person-threshold` এবং raw keypoint heatmap score cutoff `--keypoint-score-threshold` আলাদা settings। কোনো match না হলে `pose_status: unmatched` থাকে। `--measure-alignment` যোগ করলে matched region center-এর crosshair offsetও JSON-এ আসে। Raw keypoint score probability নয়। Alignment offset hit probability বা accuracy নয়। Dashboard body-region center ও alignment offset-ও দেখায়।

`vd-1` frame 3000, 3030 ও 3060-এ তিন-frame smoke run-এ 10টি person candidate ছিল। Frame 3000-এ 2টি candidate-এর সঙ্গে pose model মেলানো গেছে; frame 3030 ও 3060-এর candidate-গুলোর কোনো pose match আসেনি। এটি model-এর intermittent miss দেখায়, accuracy metric নয়। ছোট sample-এ full video performance বা body-region accuracy দাবি করা যাবে না।

Integration smoke run-এ frame 3000-এর 3টি person candidate-এর জন্য 2টি pose observation match হয়েছে এবং 8টি coarse region alignment measurement JSON-এ লেখা হয়েছে। এক frame-এ combined throughput `0.325` sampled frame/s ছিল (detector mean `1170.7ms`, pose mean `893.3ms`); এটি শুধু একটি frame ও এই GPU run-এর সময়, representative speed benchmark নয়। Matched pose-গুলোর মধ্যে false person candidate থাকতে পারে; hand labels ছাড়া region correctness জানা যায় না।

Frame 3000-এর combined preview-তে একটি in-game character candidate-এর পাশাপাশি webcam/first-person এলাকার ওপর বড় person candidate এবং pose keypoint-ও দেখা গেছে। এগুলো likely overlay/weapon false positive, কিন্তু হাতে label না করা পর্যন্ত evaluator-এ false positive হিসেবে গণনা করছি না। Webcam-এর জন্য rectangle বড় করলে সেই অংশের সম্ভাব্য in-game target-ও বাদ পড়তে পারে; তাই frame review ছাড়া mask বড় করা হয়নি।

### M8: আলাদা offline dashboard

Saved JSON report ও তার source video দেখতে:

```powershell
python tools/open_dashboard.py reports/tracks.json
```

Dashboard sampled frame-এর তালিকা দেয়; frame নির্বাচন করলে local video থেকে ওই frame পড়ে JSON report-এর candidate box/temporary ID এবং থাকলে pose-region/alignment overlay আঁকে। পাশে timestamp, detector score, box coordinate ও image-plane velocity দেখা যায়। Header-এ analysis throughput ও region অনুযায়ী alignment error-এর median/p95 থাকে। `reports/detector_evaluation.csv` ও `reports/pose_region_evaluation.csv` থাকলে একই video-র precision/recall/F1 এবং region coverage/error-ও দেখায়। অন্য folder-এর video হলে `--video path\to\clip.mp4` দাও। Dashboard live screen/game capture করে না। এই environment থেকে Windows window-তে click করে visual review করা যায়নি; স্থানীয় desktop-এ চালিয়ে UI layout দেখে নিতে হবে।

### M10: sampled-frame playback control

Dashboard-এ `Play samples`, `Pause` ও `Stop` control যোগ হয়েছে। Play report-এর analyzed sample-গুলো timestamp অনুযায়ী সামনে নেয়; Pause বর্তমান sample-এ থামে; Stop প্রথম sample-এ ফেরে। Previous/Next বা frame list দিয়ে হাতে frame বাছলেও playback থেমে যায়। এটি আসল ভিডিওর প্রতিটি frame চালায় না; এটি detection ও temporary ID review করার জন্য sampled-frame sequence। UI-টি এই environment-এ হাতে চালিয়ে visual review করা হয়নি—স্থানীয় desktop-এ যাচাই করতে হবে।

### M11: dashboard থেকে local video analysis

বর্তমান dashboard-এর `Analyze a local video` panel-এ video বেছে নিয়ে প্রতি কত source frame পর analysis হবে তা নির্ধারণ করা যায়। ঐচ্ছিকভাবে body-region estimate ও frame-center alignment চালানো যায়; alignment নির্বাচন করতে body-region estimate-ও চালু থাকতে হবে। Analysis background process-এ চলে, `Cancel analysis` দিয়ে থামানো যায়, এবং সফল হলে নতুন JSON report নিজে থেকে dashboard-এ খোলে। CSV/JSON output timestamp-সহ নাম নিয়ে ignored `reports/` folder-এ লেখা হয়। এই workflow নির্বাচিত local recording পড়ে; live capture বা game control করে না। GUI flow এই environment-এ হাতে যাচাই করা যায়নি—তোমার desktop-এ নতুন video দিয়ে যাচাই দরকার।

### M12: provisional video evaluation

Frame `0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 6000, 6600`—প্রতি 5s-এ একটি—local preview/source frame দেখে review করা হয়েছে। 10s, 20s, 25s ও 45s positive; 0s, 5s, 35s ও 50s negative; 15s, 30s, 40s ও 55s uncertain রাখা হয়েছে। Positive box-গুলো original 2560×1440 frame coordinate-এ: frame 1200 `[1575,195,1760,625]`, 2400 `[1195,710,1267,866]`, 3000 `[1247,505,1350,726]`, 5400 `[1085,675,1175,880]`। Annotation ও CSV Git-ignored local data। Confidence `0.25`, IoU `0.50`-এ 8 scored frame থেকে `TP=3, FP=24, FN=1`, precision `0.111`, recall `0.750`, F1 `0.194` হয়েছে। Frame 2400-এ person candidate-এর box এত বড় ছিল যে visible target-এর overlap IoU `0.50`-এ পৌঁছায়নি; তাই সেটি miss হিসেবে গণনা হয়েছে। এটি assistant-এর provisional visual review, user-verified benchmark নয়। Box ও উপস্থিতি-label user annotation window-তে যাচাই/সংশোধন করে আলাদা `vd-1_user_verified.jsonl` file দিয়ে evaluator rerun করতে হবে।

User-verified copy বানাতে:

```powershell
python tools/annotate_video_frames.py "data/input/videos/vd-1.mp4" --frames 0,600,1200,1800,2400,3000,3600,4200,4800,5400,6000,6600 --prefill data/annotations/vd-1_provisional_review.jsonl --output data/annotations/vd-1_user_verified.jsonl
python tools/evaluate_detector.py "data/input/videos/vd-1.mp4" data/annotations/vd-1_user_verified.jsonl --expected-frames 0,600,1200,1800,2400,3000,3600,4200,4800,5400,6000,6600 --confidence-threshold 0.25 --iou-threshold 0.50 --output reports/vd-1_user_verified_eval.csv
```

Evaluator default-এ শুধু interactive annotation tool-এর manually reviewed row গ্রহণ করে। পুরোনো/প্রাথমিক provisional file দিয়ে কেবল exploratory comparison করতে চাইলে স্পষ্টভাবে `--allow-unreviewed-labels` দাও; output CSV-তে `unreviewed_exploratory` provenance ও terminal-এ warning থাকবে। উদাহরণ:

```powershell
python tools/evaluate_detector.py "data/input/videos/vd-1.mp4" data/annotations/vd-1_provisional_review.jsonl --allow-unreviewed-labels --confidence-threshold 0.25 --iou-threshold 0.50 --output reports/vd-1_provisional_eval.csv
```

### M13: confidence-threshold sensitivity

একই 8টি provisional scored frame-এ শুধু confidence cutoff বদলে evaluation করা হয়েছে:

| Confidence | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 3 | 24 | 1 | 0.111 | 0.750 | 0.194 |
| 0.50 | 2 | 16 | 2 | 0.111 | 0.500 | 0.182 |
| 0.75 | 2 | 10 | 2 | 0.167 | 0.500 | 0.250 |

Cutoff `0.75`-এ false positive কমলেও recall অর্ধেকে নেমেছে; frame 5400-এর ছোট/distant target `0.50` ও `0.75`-এ বাদ পড়েছে। Frame 2400-এর oversized box higher confidence-এ থেকেও IoU `0.50`-এ match হয়নি—শুধু threshold বাড়িয়ে localization সমস্যা মেটে না। এই অতি ছোট provisional sample দেখে default `0.25` বদলানো হয়নি। CSV-গুলো `reports/vd-1_provisional_eval_conf_0_50.csv` ও `_conf_0_75.csv` নামে local, Git-ignored output।

25.0–25.5s sample-এ visible opponent মাত্র 2টি sampled frame পর্যন্ত একই ID পেয়েছে, এরপর detector তাকে miss করেছে। First-person/overlay false candidate-ও track হয়। IoU tracker দ্রুত movement, box-shape পরিবর্তন ও missed detection-এ ID fragment করতে পারে; ফলে এটি baseline association experiment, validated tracking নয়।

### M14: dashboard থেকে screenshot ও video input

Dashboard-এর `Choose image or video…` button দিয়ে local PNG/JPG/BMP/WebP screenshot অথবা video বেছে নেওয়া যায়। Analysis শুরুর আগে screenshot-এর dimensions, বা video-র dimensions/FPS/duration দেখায়। Screenshot-এর জন্য এক-frame person-candidate analysis চলে; video-র জন্য আগের frame-step sampled tracking চলে। দুটোতেই ঐচ্ছিক body-region estimate ও frame-center alignment আছে। কাজ শেষ হলে dashboard নতুন report খোলে। Screenshot report-এ temporary ID কেবল ওই ছবির ভেতরের candidate order; এটি tracking নয়।

Screenshot report CLI দিয়ে খুললে source default হিসেবে `data/input/screenshots/`-এ খোঁজা হয়; অন্য path হলে `--image` দিতে হবে. Direct run:

```powershell
python tools/analyze_image.py "data/input/screenshots/ss-1.png" --estimate-body-regions --measure-alignment --output reports/ss-1.csv --json-output reports/ss-1.json
python tools/open_dashboard.py reports/ss-1.json
```

`ss-1.png`-এ বর্তমান local environment-এ detector ও pose estimate চালিয়ে JSON report schema loader দিয়ে যাচাই করা হয়েছে: 1917×1073 screenshot-এ 1টি COCO person candidate এসেছে। এটি detector correctness বা enemy identity-র প্রমাণ নয়। Desktop GUI-তে হাতে click করে layout review এখনো করা হয়নি। নতুন CLI-তে file picker থেকে metadata এবং image/video report reopen flow এই desktop session-এ end-to-end যাচাই বাকি।

### M15: provisional box manually review করার সহায়তা

Annotation tool-এ `--prefill` যোগ হয়েছে। এটি provisional JSONL-এর box-গুলো frame-এ suggestion হিসেবে দেখায়, কিন্তু label pre-select করে না। `y`, `n` বা `u` তোমাকেই চাপতে হবে; তারপর `Enter` দিলে আলাদা output-এ save হবে। ভুল/বাড়তি suggestion সরাতে `c` দিয়ে সব box clear করে সঠিক box নতুন করে আঁকো, অথবা দৃশ্যে opponent না থাকলে clear করে `n` দাও। Ambiguous হলে `u` নাও। প্রতিটি saved row-তে `reviewed_manually: true` ও prefill ব্যবহারের indicator থাকবে।

`vd-1`-এর 12টি sample যাচাই করতে:

```powershell
python tools/annotate_video_frames.py "data/input/videos/vd-1.mp4" --frames 0,600,1200,1800,2400,3000,3600,4200,4800,5400,6000,6600 --prefill data/annotations/vd-1_provisional_review.jsonl --output data/annotations/vd-1_user_verified.jsonl
```

একটি করে frame ভালোভাবে দেখে status ও box নিশ্চিত করে Enter দাও। `q` দিলে ইতিমধ্যে save হওয়া row-গুলো থাকবে; একই command পুনরায় চালালে সেগুলো skip হবে। Output Git-ignored। এই review window এখনো user-এর হাতে চালানো বাকি।

### M16: চলমান video analysis pause/resume

Dashboard-এ video analysis চলাকালে `Pause video analysis` চাপলে থামবে, `Resume video analysis` চাপলে আবার চলবে, আর `Cancel analysis` পুরো run বন্ধ করবে। Pauseটি video-এর sampled-frame boundary-তে কার্যকর; paused সময়ে model inference চলে না। JSON report-এর `analysis_paused_seconds`-এ user-এর pause time থাকে; `analysis_elapsed_seconds` ও analysis throughput হিসাব থেকে ওই অপেক্ষার সময় বাদ যায়। Screenshot analysis এক-frame atomic inference হওয়ায় pause button সক্রিয় হয় না। GUI হাতে চালিয়ে button যাচাই করা এখনো বাকি।
`vd-1.mp4`-এর 3 sampled frame (`0–60`, step 30) দিয়ে local CUDA smoke run-এ pause sentinel 21.31s রাখা হয়েছিল। Run শেষে 3 frame complete হয়েছে; report-এ paused `21.31s`, active elapsed `4.04s`, throughput `0.742 sampled frame/s` ছিল—pause সময় বাদ পড়েছে। এটি pause accounting-এর implementation check; 3 frame-এর throughput performance benchmark নয়। `25` unit test pass, `compileall` ও `git diff --check` clean. CUA app inventory-তে Tkinter dashboard window expose হয়নি, তাই button-গুলোর visual/manual interaction যাচাই বাকি।

### M17: user-labeled screenshot presence evaluation

`data/annotations/screenshot_presence.csv`-এর user-labeled yes/no তথ্যের সঙ্গে প্রতিটি screenshot-এ অন্তত একটি person candidate পাওয়া গেছে কি না তুলতে:

```powershell
python tools/evaluate_screenshot_presence.py --confidence-threshold 0.25
```

Tool প্রতি screenshot-এর candidate count ও সর্বোচ্চ score-সহ ignored `reports/screenshot_presence_eval.csv` লেখে এবং TP/FP/TN/FN, accuracy, precision, recall, specificity, F1 দেখায়। বর্তমান দুই screenshot-এ local CUDA run-এ `ss-1`-এ candidate পাওয়া গেছে এবং `ss-2`-এও একটি false candidate পাওয়া গেছে: `TP=1, FP=1, TN=0, FN=0`; accuracy `0.500`, precision `0.500`, recall `1.000`, specificity `0.000`, F1 `0.667`। `n=2`-এর pilot। এটি box localization বা সাধারণ gameplay accuracy যাচাই করে না। `ss-2`-এর false positive detector-এর বর্তমান দুর্বলতা দেখায়। আরও screenshot label এবং user-verified video sample দরকার।

### M18: detector confidence distribution

Screenshot ও video analysis JSON report-এ সব returned person-candidate-এর detector score-এর count, min, mean, median, p95 ও max যোগ হয়েছে। কোনো candidate না থাকলে count `0`, অন্য সব পরিসংখ্যান `null` থাকে। Dashboard header median/p95 score দেখায়; report loader distribution-এর মান ও total candidate count সামঞ্জস্যপূর্ণ কি না যাচাই করে। এগুলো threshold-পেরোনো candidate score-এর বর্ণনামূলক পরিসংখ্যান—enemy identity, calibrated probability, hit probability বা detection accuracy নয়। Local CUDA smoke run-এ `vd-1`-এর 3 sampled frame-এ 17 candidate-এর median score `0.431`, p95 `0.978`, min `0.250`, max `0.978` পাওয়া গেছে; analysis throughput ছিল `0.890` sampled frame/s। `ss-1`-এ 1 candidate-এর score `0.286`। দুই report-ই schema loader-এ যাচাই হয়েছে। অতি ছোট run-এর throughput hardware benchmark নয়। 32 unit test pass, compileall ও diff check clean।

### M19: evaluation-এর আগে annotation provenance gate

Detector ও pose evaluator এখন default-এ কেবল interactive annotation tool দিয়ে হাতে review করে save করা row গ্রহণ করে (`reviewed_manually: true`, provisional note ছাড়া)। Assistant-এর provisional label বা marker-বিহীন পুরোনো JSONL default-এ ground truth হিসেবে চলবে না। কেবল exploratory comparison-এর জন্য `--allow-unreviewed-labels` দিতে হবে; output-এ `unreviewed_exploratory` provenance লেখা হবে এবং warning দেখাবে। এতে পুরোনো `vd-1_provisional_review.jsonl` হারায় না, কিন্তু সেটি user-verified বলে ভুল উপস্থাপন হয় না। যাচাইয়ে default run provisional file reject করেছে এবং কোনো CSV লেখেনি; explicit exploratory run 8 scored row লিখেছে, প্রত্যেকটিতে `unreviewed_exploratory`, metric আগের provisional ফলের সঙ্গে মিলে `TP=3, FP=24, FN=1` হয়েছে। 39 unit test pass, compileall ও diff check clean।

### M20: পুরো `vd-1` clip-এ 1-second sampled tracking

59-second, 120 FPS video-তে প্রতি 120 source frame-এ একটি sample চালানো হয়েছে:

```powershell
python tools/track_video.py "data/input/videos/vd-1.mp4" --frame-step 120 --max-side 1920 --output reports/vd-1_coarse_1s_tracking.csv --json-output reports/vd-1_coarse_1s_tracking.json
```

59টি sampled frame-এ 188 person-candidate observation এসেছে; 11 frame-এ কোনো candidate আসেনি। 127 temporary ID-এর মধ্যে 96টি ID এক sample-এ দেখা গেছে; কোনো ID সর্বোচ্চ 6টি sample-এ candidate জুড়েছে। Detector latency mean `145.8ms`, p95 `195.7ms`; observed sampled analysis throughput `1.016 frame/s`। নতুন JSON report 59 frame-সহ dashboard report loader-এ validate হয়েছে। সব CSV/JSON output `reports/`-এ Git-ignored। এগুলো detector candidate ও sparse IoU association-এর পরিমাপ—enemy accuracy বা tracking continuity ground truth নয়; ID-গুলো player identity-ও নয়। Throughput এই machine/model/config-এ মাপা, অন্য hardware-এর benchmark নয়।


### M21: sampled video-তে indexed seeking

M20-তে frame-step `120` হলেও আগের loop 7,080 source frame sequentially decode করে 59টি inference sample নিয়েছিল। Input reader-এ `sampled_frames(frame_step, start_frame, stop_frame)` যোগ করে প্রতিটি দরকারি index সরাসরি seek করা হয়েছে; `--stop-frame` আগের মতো exclusive। Synthetic MJPG test-এ index `2,5,8`, timestamp ও decoded pixel পরীক্ষা pass করেছে। `vd-1`-এর একই `59` frame পুনরায় চালালে candidate output হুবহু একই ছিল (`188` candidates), কিন্তু throughput `1.016` থেকে `3.207 sampled frame/s` হয়েছে—এই machine ও video-তে প্রায় `3.2×`। Mean detector latency `139.8ms`, p95 `202.9ms`; report loader-এ 59-frame JSON validate হয়েছে। এটি শুধু ওই hardware/clip/settings-এর মাপ, detector accuracy বা real-time benchmark নয়। Unit test suite `41/41`, `compileall` ও diff check pass করেছে।
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

M1–M3-এর baseline, M5 body-region experiment, M6 sampled tracking, M7 alignment calculation, M8 dashboard, M9 evaluation tools, M10 sampled-frame playback, M11 dashboard video-run workflow, M12 provisional evaluation, M13 threshold sensitivity, M14 dashboard image/video input, M15 provisional-box review flow, M16 video analysis pause/resume, M17 screenshot presence evaluation, M18 confidence distribution, M19 annotation provenance gate, M20 full-clip coarse tracking, M21 indexed sampling, M22 body-region heatmap response summary, M23 temporary track observation summary, M24 report validation, M25 dashboard summary wrapping, M26 offline-boundary audit এবং M27 planned annotation coverage check, M28 annotation review progress indicator এবং M29 source-frame fingerprint, M30 annotation record serialization test এবং M31 confidence threshold sweep আছে। পরের কাজ হলো annotation window দিয়ে 12টি video frame user-verify করে আলাদা output file-এ রাখা, evaluator পুনরায় চালানো, তারপর বিভিন্ন সময়ের positive/negative sample যোগ করা। Body-region evaluator চালাতে visible opponent-এর box-এর ভেতরে head/chest/lower-torso/legs point-ও label করতে হবে। User-reviewed label ও বৈচিত্র্যময় sample না পাওয়া পর্যন্ত detector score final accuracy নয়।

## ভাষার নিয়ম

এই workflow নথিটি Bangla-তে থাকবে, যাতে ধাপগুলো অনুসরণ করা সহজ হয়। Code, variable, command ও মূল project documentation আপাতত English-এ থাকবে। Project শেষ হলে প্রয়োজন অনুযায়ী পুরো project-কে consistent international English-এ নেওয়া হবে।

### M22: body-region raw heatmap response summary

প্রতিটি estimated head/chest/lower-torso/legs region-এর `mean_keypoint_score` report-এ region অনুযায়ী count, min, mean, median, p95 ও max হিসেবে যুক্ত হয়েছে। Dashboard header median/p95 দেখায়। এটি pose model-এর raw heatmap response মাত্র; calibrated confidence বা সঠিক body point পাওয়ার probability নয়, এবং মান 1-এর বেশি হতে পারে। ফলে detector candidate confidence-এর সাথে এক করে পড়া যাবে না।

`ss-1.png`-এ একটি person candidate থেকে চারটি region estimate পাওয়া গেছে; observed raw response head `2.055`, chest `0.319`, lower torso `3.099`, legs `3.544`। `vd-1`-এর frame 0–120-এ প্রতি 60 frame-এ 3টি sampled frame চালিয়ে 26 candidate পাওয়া গেছে; response summary chest/head-এ 9টি করে, lower torso-তে 7টি, legs-এ 6টি available estimate দেখিয়েছে। দুই report-ই dashboard report loader-এ valid হয়েছে। এটি ছোট sample এবং কোনো ground-truth accuracy পরীক্ষা নয়। Unit test suite `42/42` pass করেছে।

### M23: temporary track observation summary

Video JSON report-এ temporary ID-গুলোর per-ID sampled observation count এবং প্রথম/শেষ observed sample-এর সময়ের ব্যবধান যোগ হয়েছে; dashboard header track count, এক-sample ID সংখ্যা এবং median/max observed span দেখায়। এটি sparse sampled observation-এর বর্ণনা। Sample-এর মাঝখানে object অবিরত ছিল, ID একই player-এর, বা tracking সঠিক—এমন দাবি করে না।

`vd-1`-এর 120-frame interval-এ full clip চালিয়ে 59 sampled frame ও 188 candidate observation পাওয়া গেছে। 127 temporary ID-এর মধ্যে 96টি এক sample-এ ছিল। Observations per ID median `1`, max `6`; observed span median `0s`, p95 `3s`, max `7s`। Median zero হওয়া মানে অর্ধেকের বেশি ID-কে একবারই দেখা গেছে। Report loader-এ JSON validate হয়েছে; unit suite `43/43`, compileall ও diff check pass করেছে। Throughput `3.415 sampled frame/s` এই machine ও run-এর measurement, detector accuracy বা অন্য computer-এর benchmark নয়।

### M24: নতুন summary field report validation

Dashboard report loader এখন M22 body-region response summary এবং M23 temporary-track summary-এর structure, finite values, count ও nonnegative time span যাচাই করে। ভুল region, distribution-এর count mismatch, অথবা track count candidate count-এর চেয়ে বেশি হলে report reject হবে। পুরোনো report-এ এই optional field না থাকলেও load হয়। Unit test suite `44/44`; M22/M23-এর তিনটি saved image/video report validate, compileall ও diff check pass করেছে।

### M25: dashboard summary header wrapping

Dashboard-এর এক লাইনের summary-তে M22/M23 যোগ হওয়ার পর metrics সরু window-তে কেটে যেতে পারত। Summary-কে source/run stats, confidence, region response ও tracking আলাদা লাইনে সাজিয়ে 1200px wrap limit দেওয়া হয়েছে। এতে dashboard-এর default 1280px window-তে দীর্ঘ raw heatmap summary পড়া সহজ হওয়ার কথা; Windows UI-তে সরাসরি visual QA এখনও বাকি।

### M26: offline-only boundary code review

`src/` ও `tools/`-এ input/capture API পর্যালোচনা করে দেখা হয়েছে video reader ও dashboard-এর `cv2.VideoCapture` নির্বাচিত file path খোলে; screen capture, game-process/memory access, injected overlay, mouse/keyboard game-input automation বা socket/request-based live stream handling পাওয়া যায়নি। `cv2.waitKey` কেবল offline annotation window-এর label key পড়তে ব্যবহৃত। Model checkpoint প্রথমবার TorchVision থেকে নামতে পারে—একবার cache হলে footage analysis selected local file-এ চলে। এটি source-code search ও path review, formal security audit নয়।

### M27: planned annotation coverage check

`evaluate_detector.py`-তে ঐচ্ছিক `--expected-frames` যুক্ত হয়েছে। Comma-separated source frame index-এর set ঠিক মিলতে হবে; কোনো planned frame missing বা বাড়তি frame থাকলে detector inference-এর আগেই evaluator স্পষ্ট error দেবে। এটি user-এর label status বা box সঠিক কি না যাচাই করে না; কেবল dataset coverage gate। `vd-1`-এর 12-frame review-এর জন্য exact list-সহ command উপরের M12 workflow-তে দেওয়া আছে। Test-এ parse validation, missing ও unexpected index যাচাই করা হয়েছে।

### M28: annotation review progress indicator

Interactive annotation window ও terminal-এ planned sample-এর `বর্তমান/মোট` progress দেখায়; আগে save করা frame skip হলেও তার sample position অক্ষুণ্ণ থাকে। এতে 12-frame review-তে কতদূর হয়েছে বোঝা সহজ হয়, কিন্তু কোনো label নিজে থেকে নির্বাচন বা পরিবর্তন হয় না। নতুন behavior window restart করে visual check করা বাকি।

### M29: source-frame fingerprint দিয়ে label মিল যাচাই

নতুন manual video annotation-এ decoded frame-এর shape, dtype ও pixel bytes থেকে SHA-256 fingerprint রাখা হয়। Detector ও pose evaluator annotation-এ fingerprint থাকলে একই frame পুনরায় decode করে hash মেলায়; filename/frame index একই রেখে source video বদলালে evaluation থেমে যায়। আগের annotation-এ fingerprint না থাকলেও compatibility-র জন্য সেগুলো গ্রহণ করা হয়। Synthetic array test-এ একই pixels স্থিতিশীল hash দিয়েছে, pixel বা shape বদলালে hash বদলেছে; match/mismatch ও malformed hash যাচাই হয়েছে। Generated MJPG clip থেকে একই index আবার decode করে hash match এবং পাশের frame-এ mismatch integration test-এ নিশ্চিত হয়েছে। Unit suite `50/50`, compileall ও diff check pass করেছে। বর্তমান frame-review window M29-এর আগের process, তাই এখন তৈরি হওয়া label-এ hash নাও থাকতে পারে; নতুন annotation run-এ hash লেখা হবে।

### M30: annotation record serialization test

GUI থেকে আলাদা `_build_annotation_record` helper manually chosen status, opponent box, optional body-region point, provenance ও source-frame hash JSONL row-তে ঠিকভাবে সাজায় কি না test করা হয়েছে। Synthetic `VideoFrame`-এ frame index/timestamp ও 192×108 dimensions, SHA-256, box এবং head/chest point যাচাই হয়েছে। এটি GUI interaction বা disk-এ JSONL append-এর visual test নয়। Unit suite `51/51`, compileall ও diff check pass করেছে।

### M31: এক inference pass-এ confidence threshold sweep

`evaluate_threshold_sweep.py` প্রতিটি scored frame-এ detector একবার চালিয়ে confidence score ও box reuse করে `.25`–`.95` cutoff-এ TP/FP/FN, precision/recall/F1 হিসাব করে CSV লেখে। একই 12টি provisional frame দিয়ে exact coverage gate এবং `--allow-unreviewed-labels`সহ run হয়েছে; 4 uncertain বাদে 8 scored frame, 15 cutoff পাওয়া গেছে। Provenance `unreviewed_exploratory` লেখা হয়েছে।

এই ছোট provisional sample-এ `.25`-এ `TP=3, FP=24, FN=1`, precision `0.111`, recall `0.750`, F1 `0.194`; `.95`-এ `TP=2, FP=4, FN=2`, precision `0.333`, recall `0.500`, F1 `0.400`। উচ্চ cutoff-এ false positive কমেছে, কিন্তু recall কমেছে; `.95`-এর F1-কে tuning recommendation ধরা যাবে না। User-verified labels এলে sweep পুনরায় করতে হবে। Threshold parser ও cached-prediction metric-এর unit testসহ suite `54/54`, compileall, CLI help ও diff check pass করেছে।
