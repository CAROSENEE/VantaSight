# VantaSight: কাজের ধাপ ও বর্তমান অবস্থা

এটি VantaSight তৈরির ধাপে ধাপে কাজের তালিকা। প্রতিটি ধাপ শুরুর আগে কী করতে যাচ্ছি তা জানাব। ধাপ শেষ হলে কী বদলেছে, কেন দরকার, কীভাবে ব্যবহার করতে হবে এবং কোনো সীমাবদ্ধতা আছে কি না—সহজ ভাষায় বুঝিয়ে দেব। কোনো কাজ যাচাই করা না গেলে সেটিও পরিষ্কার করে বলব।

**Annotation status (2026-11-22):** আগের OpenCV annotation attempt-এ process frame 540-এ চললেও user window দেখেননি; তাই সেটি annotation ছিল না। পরে stale OpenCV process বন্ধ করা হয়েছে। Local browser review gallery-তে ১২টি frame, opponent box, optional body-region point ও zoom controls প্রস্তুত; কিন্তু `vd-1_user_verified.jsonl` এখনো নেই এবং কোনো video label save/confirm হয়নি। তাই verified video ground truth বা detector/pose accuracy metric নেই; annotation user review ও label-নির্ভর evaluation pending। আগের unverified provisional label ও derived local reports মুছে ফেলা হয়েছে।

## প্রকল্পের লক্ষ্য

প্রথমে Valorant-এর নিজের screenshot বা recorded gameplay video বিশ্লেষণ করে দৃশ্যমান তথ্য নিয়ে গবেষণা করা: character candidate শনাক্ত করা, দৃশ্যমান body-region অনুমান করা, পরপর frame-এ detection track করা, এবং crosshair reference থেকে target region-এর দূরত্ব মাপা। ভবিষ্যতে অন্য FPS বা online game-এ নেওয়ার উপযোগী করে input ও analytics-এর সাধারণ অংশগুলো game-specific অংশ থেকে আলাদা রাখা হবে। প্রকল্পটি game input পাঠাবে না, aim নড়াবে না, fire করবে না, game memory পড়বে না বা game process-এ প্রবেশ করবে না। প্রথম সংস্করণ শুধু offline footage বিশ্লেষণ করবে।

## এখন পর্যন্ত কী করা হয়েছে

- **M1 — Input foundation:** image loader, video metadata reader ও CLI লেখা এবং `vd-1.mp4` দিয়ে runtime-এ যাচাই করা হয়েছে।
- **M2 — Frame preview:** নির্দিষ্ট video frame পড়া ও preview image save করা হয়েছে। Frame index 3540 (29.5 সেকেন্ড) সফলভাবে বের হয়েছে: `data/previews/vd-1_frame_003540.png`। ছয়টি অতিরিক্ত সময়ে frame দেখে contact sheet বানানো হয়েছে: `data/previews/vd-1_contact-sheet.jpg`।
- **M3 — Baseline detection:** TorchVision COCO `person candidate` detector যোগ করে `ss-1`, `ss-2` এবং `vd-1`-এর frame 3000-এ runtime inference করা হয়েছে।
- `ss-1`, `ss-2` ও `vd-1`-এ baseline detector/region model চালিয়ে candidate preview তৈরি হয়েছিল। এগুলো model output; enemy identity বা user-verified detection label নয়.
- আগে সংরক্ষিত “সঠিক/ভুল candidate” hand-review CSV এবং threshold comparison-কে verified ground truth হিসেবে ধরা যাবে না; সংশ্লিষ্ট local label CSV মুছে ফেলা হয়েছে।
- **M5 — Body-region experiment:** TorchVision Keypoint R-CNN-এর COCO_V1 দিয়ে `ss-1`, `vd-1` frame 3000 (25s) ও frame 5400 (45s)-এ GPU inference হয়েছে। `ss-1`-এ 1টি, `vd-1` frame 3000-এ 2টি এবং frame 5400-এ 0টি person candidate-এ pose estimate এসেছে। এগুলো model output; কোন candidate enemy বা body-region estimate সঠিক—তা label দিয়ে যাচাই হয়নি.
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
- দেওয়া দুটি screenshot `ss-1.png` ও `ss-2.png` নামে input folder-এ রাখা হয়েছে; resolution যথাক্রমে 1917×1073 এবং 1917×1070। এগুলো ও screenshot presence-এর দুইটি user-provided label reproducible pilot sample হিসেবে GitHub-এ রাখা হয়েছে।
- প্রাথমিক visual label: `ss-1`-এ gameplay doorway-তে enemy character দৃশ্যমান; `ss-2`-এ main sightline-এ সাধারণ gameplay character দৃশ্যমান নয়। `ss-2`-এর লাল replay silhouette-গুলোকে আলাদা overlay হিসেবে ধরব, normal gameplay player detection-এর ground truth হিসেবে নয়।
- Video ও screenshot-এর ID আলাদা ধারায় চলবে: `vd-1`, `vd-2`, ... এবং `ss-1`, `ss-2`, ...। File extension আগের মতো থাকবে, যেমন `vd-1.mp4` বা `ss-1.png`।
- `README.md`-তে setup ও ব্যবহারের নির্দেশনা আছে। `.gitignore` সাধারণ input media ও generated preview বাদ রাখে; নির্বাচিত reproducibility sample আলাদা করে track করা।
- Detector-এর weight source, license note ও known limitation `docs/MODEL_CARDS.md`-এ লেখা আছে।
- Screenshot-এর user-provided positive/negative label `data/annotations/screenshot_presence.csv`-এ রাখা হয়েছে। এই pilot CSV ও `source_regions.json` tracked; পরবর্তী video annotation ও generated report Git ignore থাকে।
- M85-এ দুইটি original screenshot ও তাদের presence labels GitHub-এ যোগ হয়েছে। `vd-1.mp4` Git LFS-এ track করা; M95-এ fresh clone থেকে download ও checksum verify হয়েছে।

Enemy box label করার OpenCV tool `tools/annotate_video_frames.py`, browser-based offline gallery `tools/export_review_gallery.py`, এবং label থেকে metric হিসাবের evaluator আছে। আগের OpenCV process বন্ধ; user window দেখতে পাননি। Browser gallery-তে ১২টি original-resolution review frame প্রস্তুত, কিন্তু কোনো label নেই। তাই verified video benchmark নেই; পুরোনো unverified metric প্রত্যাহার ও local data থেকে মুছে ফেলা হয়েছে। Body-region estimate-ও manual point label দিয়ে যাচাই হয়নি।

## কাজের roadmap

| ধাপ | কাজ | ধাপ শেষে যা পাওয়া যাবে | অবস্থা |
|---|---|---|---|
| 1. Input foundation | Python package, screenshot/video reader, metadata | Image ও video load করার ভিত্তি | `vd-1.mp4` দিয়ে runtime-এ যাচাই হয়েছে |
| 2. Frame preview | Video থেকে নির্দিষ্ট frame বের করা ও preview image save করা | বাস্তব footage-এর নির্দিষ্ট frame দেখা | Frame 3540 সফলভাবে বের ও দেখা হয়েছে |
| 3. প্রাথমিক detection | COCO pretrained person-candidate model; confidence ও bounding box দেখানো | Model কোন জিনিস শনাক্ত করছে তা পর্যালোচনা | User-labeled 2-screenshot pilot-এ default 0.25 cutoff-এ TP=1, FP=1; video/general performance এখনো অপ্রমাণিত |
| 4. Dataset review | বাস্তবসম্মত clip-এ false positive ও missed detection হাতে পর্যালোচনা | Detector উপযোগী কি না তার প্রাথমিক প্রমাণ | OpenCV window ও local HTML gallery প্রস্তুত; ১২ frame-এ user label বাকি, verified video metric নেই |
| 5. Body-region পরীক্ষা | Pose/region model চেষ্টা করা ও অনিশ্চয়তা দেখানো | Score ও uncertainty-সহ region estimate | Screenshot pilot-এ 31×49 px target matched, arm false candidate unmatched; মাত্র n=2, video/manual-point validation বাকি |
| 6. Tracking | একাধিক frame-এর detection মিলিয়ে temporary ID দেওয়া | Track কতটা ধারাবাহিক তা দেখা | IoU tracker, CSV/JSON export আছে; 59-frame coarse sample-এ 127 candidate track-এর 96টি এক observation-এ শেষ হয়েছে; continuity review বাকি |
| 7. Alignment metrics | Crosshair reference সেট করা এবং pixel/normalized distance হিসাব | জ্যামিতিক alignment measurement | `ss-1` ও `vd-1` frame 3000-এ runtime check হয়েছে; geometry-only, region label দিয়ে যাচাই বাকি |
| 8. Export ও dashboard | CSV/JSON এবং আলাদা analysis preview/dashboard | ব্যবহারযোগ্য research prototype | CSV/JSON export, body-region/alignment overlay, detector/pose summary, confidence sweep ও IoU/AP curve-সহ dashboard আছে; Windows launcher যোগ হয়েছে, পূর্ণ UI visual review বাকি |
| 9. Evaluation ও documentation | Annotated sample-এ metric মাপা, limitation লেখা | পুনরায় যাচাইযোগ্য ফল ও setup guide | Evaluator code আছে; screenshot label আলাদা, কিন্তু video ground truth/valid video metric ও pose-region evaluation বাকি |

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
python tools/create_review_shortlist.py reports/vd-1_dashboard_tracks_dense.json --output reports/vd-1_manual_review_shortlist.csv
python tools/annotate_video_frames.py "data/input/videos/vd-1.mp4" --shortlist-csv reports/vd-1_manual_review_shortlist.csv --dry-run
python tools/annotate_video_frames.py "data/input/videos/vd-1.mp4" --shortlist-csv reports/vd-1_manual_review_shortlist.csv
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

Dashboard sampled frame-এর তালিকা দেয়; frame নির্বাচন করলে local video থেকে ওই frame পড়ে JSON report-এর candidate box/temporary ID এবং থাকলে pose-region/alignment overlay আঁকে। পাশে timestamp, detector score, box coordinate ও image-plane velocity দেখা যায়। Header-এ analysis throughput ও region অনুযায়ী alignment error-এর median/p95 থাকে। Detector/pose evaluation CSV default-এ load হয় না; user-verified label report থাকলে আলাদা করে `--detector-evaluation` বা `--pose-evaluation` দিয়ে দিতে হয়। Confidence sweep ও IoU/AP report-ও explicit path দিয়ে দিতে হয়। অন্য folder-এর video হলে `--video path\to\clip.mp4` দাও। Dashboard live screen/game capture করে না। এই environment থেকে Windows window-তে click করে visual review করা যায়নি; স্থানীয় desktop-এ চালিয়ে UI layout দেখে নিতে হবে।

### M10: sampled-frame playback control

Dashboard-এ `Play samples`, `Pause` ও `Stop` control যোগ হয়েছে। Play report-এর analyzed sample-গুলো timestamp অনুযায়ী সামনে নেয়; Pause বর্তমান sample-এ থামে; Stop প্রথম sample-এ ফেরে। Previous/Next বা frame list দিয়ে হাতে frame বাছলেও playback থেমে যায়। এটি আসল ভিডিওর প্রতিটি frame চালায় না; এটি detection ও temporary ID review করার জন্য sampled-frame sequence। UI-টি এই environment-এ হাতে চালিয়ে visual review করা হয়নি—স্থানীয় desktop-এ যাচাই করতে হবে।

### M11: dashboard থেকে local video analysis

বর্তমান dashboard-এর `Analyze a local video` panel-এ video বেছে নিয়ে প্রতি কত source frame পর analysis হবে তা নির্ধারণ করা যায়। ঐচ্ছিকভাবে body-region estimate ও frame-center alignment চালানো যায়; alignment নির্বাচন করতে body-region estimate-ও চালু থাকতে হবে। Analysis background process-এ চলে, `Cancel analysis` দিয়ে থামানো যায়, এবং সফল হলে নতুন JSON report নিজে থেকে dashboard-এ খোলে। CSV/JSON output timestamp-সহ নাম নিয়ে ignored `reports/` folder-এ লেখা হয়। এই workflow নির্বাচিত local recording পড়ে; live capture বা game control করে না। GUI flow এই environment-এ হাতে যাচাই করা যায়নি—তোমার desktop-এ নতুন video দিয়ে যাচাই দরকার।

### M12: video annotation — এখনো সম্পন্ন হয়নি

কোনো annotation window user-এর কাছে দেখা যায়নি; কোনো video frame-এর label-ও user confirm করেননি। আগের provisional rows এবং সেগুলো থেকে বানানো detector metric তাই ground truth হিসেবে বাতিল; unverified label ও derived report local folder থেকে মুছে ফেলা হয়েছে। Annotation tool-এর code বা command থাকা মানে annotation হওয়া নয়। User আগে review পরে করতে বলেছেন, তাই frame review বা evaluator চালানো হয়নি।

Windows-এ browser gallery নিজে খুলতে `tools/start_vd1_review_gallery.bat` double-click করুন। এটি current template দিয়ে gallery refresh করে তারপর browser খোলে। Frame দেখে status/box এবং চাইলে region point নিজে save করুন। Download করা JSONL validate করে canonical path-এ আনতে M98 importer ব্যবহার করুন।

OpenCV annotation window ব্যবহার করতে চাইলে blank review preflight ও launcher command:

```powershell
python tools/annotate_video_frames.py "data/input/videos/vd-1.mp4" --shortlist-csv reports/vd-1_manual_review_shortlist.csv --dry-run
python tools/annotate_video_frames.py "data/input/videos/vd-1.mp4" --shortlist-csv reports/vd-1_manual_review_shortlist.csv --output data/annotations/vd-1_user_verified.jsonl
```

Frame ধরে দেখে user নিজে visible opponent-এর box/status নিশ্চিত করে save করবেন। Browser বা OpenCV—কোনো tool চলা মানেই label save হয়েছে নয়; verified output user observation ও Save action ছাড়া তৈরি হয়েছে বলে ধরা যাবে না। Evaluator কেবল user review শেষ হওয়ার পর আলাদা ধাপে চালানো হবে।
### M13: confidence-threshold sensitivity — বাতিল

আগের provisional annotation দিয়ে তৈরি threshold table ও সংশ্লিষ্ট detector metric বাতিল। Annotation window user দেখেননি এবং label নিশ্চিত করেননি; তাই ওই সংখ্যাগুলো ground truth-এর সঙ্গে তুলনা নয় এবং detector performance-এর প্রমাণ নয়। সংশ্লিষ্ট local provisional label ও derived report মুছে ফেলা হয়েছে। Verified label তৈরি না হওয়া পর্যন্ত এই ধাপ অসম্পূর্ণ থাকবে; কোনো cutoff recommendation এই ফলের ওপর নির্ভর করবে না।
25.0–25.5s sample-এ visible opponent মাত্র 2টি sampled frame পর্যন্ত একই ID পেয়েছে, এরপর detector তাকে miss করেছে। First-person/overlay false candidate-ও track হয়। IoU tracker দ্রুত movement, box-shape পরিবর্তন ও missed detection-এ ID fragment করতে পারে; ফলে এটি baseline association experiment, validated tracking নয়।

### M14: dashboard থেকে screenshot ও video input

Dashboard-এর `Choose image or video…` button দিয়ে local PNG/JPG/BMP/WebP screenshot অথবা video বেছে নেওয়া যায়। Analysis শুরুর আগে screenshot-এর dimensions, বা video-র dimensions/FPS/duration দেখায়। Screenshot-এর জন্য এক-frame person-candidate analysis চলে; video-র জন্য আগের frame-step sampled tracking চলে। দুটোতেই ঐচ্ছিক body-region estimate ও frame-center alignment আছে। কাজ শেষ হলে dashboard নতুন report খোলে। Screenshot report-এ temporary ID কেবল ওই ছবির ভেতরের candidate order; এটি tracking নয়।

Screenshot report CLI দিয়ে খুললে source default হিসেবে `data/input/screenshots/`-এ খোঁজা হয়; অন্য path হলে `--image` দিতে হবে. Direct run:

```powershell
python tools/analyze_image.py "data/input/screenshots/ss-1.png" --estimate-body-regions --measure-alignment --output reports/ss-1.csv --json-output reports/ss-1.json
python tools/open_dashboard.py reports/ss-1.json
```

`ss-1.png`-এ বর্তমান local environment-এ detector ও pose estimate চালিয়ে JSON report schema loader দিয়ে যাচাই করা হয়েছে: 1917×1073 screenshot-এ 1টি COCO person candidate এসেছে। এটি detector correctness বা enemy identity-র প্রমাণ নয়। Desktop GUI-তে হাতে click করে layout review এখনো করা হয়নি। নতুন CLI-তে file picker থেকে metadata এবং image/video report reopen flow এই desktop session-এ end-to-end যাচাই বাকি।

### M15: annotation prefill — প্রত্যাহার করা হয়েছে

একসময় provisional box-কে annotation window-তে suggestion হিসেবে দেখানোর `--prefill` সুবিধা ছিল। এতে unverified computer-generated box user-এর বিচারকে প্রভাবিত করতে পারত। তাই launcher ও README command থেকে prefill বাদ দিয়ে blank review রাখা হয়েছে। এই সুবিধা ব্যবহার করে user annotation সম্পন্ন হয়েছে—এমন কোনো দাবি নেই।
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

Detector ও pose evaluator default-এ কেবল interactive annotation tool দিয়ে হাতে review করে save করা row গ্রহণ করে (`reviewed_manually: true`)। Unreviewed row default-এ ground truth হিসেবে গ্রহণ করা হয় না। পুরোনো provisional file থেকে করা exploratory run থাকলেও তার labels/metrics user-verified নয়; সংশ্লিষ্ট `vd-1` artifacts এখন মুছে ফেলা হয়েছে। তাই `vd-1`-এর জন্য কোনো ব্যবহারযোগ্য detector/pose accuracy metric নেই।

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

GitHub-এ reproducibility-র জন্য `ss-1.png`, `ss-2.png`, screenshot presence-এর user-provided CSV, sample `source_regions.json`, এবং `vd-1.mp4` Git LFS pointer যোগ করা হয়েছে। অন্য input media, generated preview/report, এবং video annotation এখনো Git ignore করা। নতুন clone-এ `vd-1.mp4` আনতে Git LFS প্রয়োজন। LFS push command সফল exit code দিয়েছে, তবে fresh clone-এ remote object fetch এখনো সম্পন্ন করে যাচাই করতে পারিনি।

Folder-গুলো খালি থাকলেও Git-এ দেখানোর জন্য ছোট `.gitkeep` placeholder file রাখা আছে। এগুলোতে ছবি বা ভিডিও থাকে না।

## বর্তমান অবস্থান (M91)

- Offline input, person-candidate detection, পরীক্ষামূলক body-region estimate, temporary tracking, alignment calculation, CSV/JSON export, dashboard, এবং manual-label evaluation tools কোডে আছে। এগুলো Valorant footage-এ ব্যবহার করা হয়েছে; detector ফল verified enemy identity বা gameplay accuracy নয়।
- M63-এ screenshot ও video analysis path-এ custom crosshair reference যোগ হয়েছে। CLI compile/help এবং বাস্তব `ss-1` ও `vd-1` inference-এ coordinate report-এ সংরক্ষিত হয়েছে।
- M64-এ dashboard-এর জন্য প্রতি 30 source frame (120 FPS-এ 0.25s) পরপর source-resolution `vd-1` report তৈরি হয়েছে; launcher এখন এই dense report খোলে। এতে ছোট/স্বল্পস্থায়ী উপস্থিতি দেখা যাওয়ার sampling chance আগের 1s sample-এর চেয়ে বেশি, তবে 0.25s-এর চেয়ে ছোট event বাদ পড়তে পারে।
- M65-এ dense report থেকে প্রতি frame-এর person-candidate box height group করার CSV tool যোগ হয়েছে; এটি annotation ছাড়াই review-এর candidate shortlist দেয়, কিন্তু recall/ground truth নয়।
- M66-এ Windows annotation launcher ও README command থেকে unverified provisional-box prefill সরানো হয়েছে; পরের user review scratch থেকে শুরু হবে।
- M67-এ desktop-এর আগে থেকে খোলা dashboard foreground-এ এনে দেখা হয়েছে: Previous/Next/Play samples/Pause/Stop controls, custom crosshair field এবং candidate-only warning দৃশ্যমান। কিন্তু window-টি পুরোনো `vd-1_dashboard_tracks.json` (59 sampled frame, 188 candidate) খুলেছে; এতে AP/threshold panel `unreviewed_exploratory` হিসেবে দেখানো। বর্তমান launcher-এর dense report এখনও চোখে দেখে খোলা হয়নি, এবং click করে playback frame পরিবর্তনও নিশ্চিত হয়নি। তাই এটি partial layout QA, playback/launcher pass নয়।
- M68-এ launcher-এ নির্ধারিত dense report ও বর্তমান local video-র SHA-256 মেলানো হয়েছে: `3917b7d5…` দুই পাশেই একই; report-এ 236 sampled frame ও 816 person candidate আছে। Viewer source mismatch-এ report reject করবে না—এটুকু নিশ্চিত, UI launch/playback বা detection accuracy নয়।
- M69-এ `reports/VantaSight Dashboard Review.lnk`-এর পুরোনো target পাওয়া গেছে: এটি `m50_tracks.json` ও unverified evaluation CSV খুলত। Local shortcut-টি এখন `tools/start_vd1_dashboard.bat` চালায়, যা dense report ব্যবহার করে। `.lnk` Git-ignored, তাই এটি শুধু এই PC-তে বদলেছে; tracked batch launcher-ই GitHub-এ আছে। Shortcut থেকে launch চেষ্টা করা হলেও নতুন window CUA-তে targetable হয়নি, তাই UI launch এখনো অসম্পূর্ণভাবে যাচাইকৃত।
- M70-এ annotation batch launcher চালুর আগে blank review ও প্রত্যাশিত window title prefix console-এ জানায়; ভুল হলে console খোলা রেখে error পড়তে বলে। এতে পরে annotation শুরু করলে কী window খুঁজতে হবে বোঝা যাবে। Annotation tool চালানো হয়নি।
- M71-এ dashboard title-এ source video এবং report filename দুটোই থাকবে; একই footage-এর পুরোনো ও dense analysis window আলাদা করে চেনা যাবে।
- M72-এ নতুন title-সহ dense dashboard সরাসরি launch করে দেখা হয়েছে। Python process চললেও CUA inventory-তে শুধু পুরোনো `VantaSight Offline Dashboard — vd-1.mp4` window ছিল; নতুন window/screenshot বা error পাওয়া যায়নি। শুধু এই attempt-এর process বন্ধ করেছি। তাই report-specific title code আছে, কিন্তু নতুন window visible/targetable হওয়া নিশ্চিত নয়।
- Video manual annotation এখনো হয়নি। M86-এ blank annotation window frame 540-এ চালু হয়েছে, কিন্তু screen visibility user-confirmed নয় এবং কোনো label save হয়নি। `vd-1_user_verified.jsonl` নেই; আগের provisional labels ও derived metrics মুছে ফেলা হয়েছে.
- পরবর্তী gate: OpenCV window বা `data/previews/vd-1_review_gallery/index.html` ব্যবহার করে ১২টি sampled frame নিজে review করে label save করা। Browser page থেকে JSONL download করে `data/annotations/`-এ রাখার পর evaluator চলবে। Dashboard dense-report playback-এর visual QA-ও এখনো বাকি।

## আমরা যেভাবে কাজ করব

1. কাজ শুরুর আগে লক্ষ্য এবং কোন file বা অংশে পরিবর্তন হতে পারে তা জানাব।
2. প্রয়োজনীয় code বা documentation update করব এবং dependency বা permission সমস্যা থাকলে জানাব।
3. ধাপ শেষে কী বদলেছে, কীভাবে ব্যবহার করতে হবে, কী যাচাই হয়েছে এবং কী সীমাবদ্ধতা আছে—তা বুঝিয়ে বলব।
4. বাস্তব footage লাগলে তোমার দেওয়া screenshot বা ছোট recorded clip দিয়ে ফল পর্যালোচনা করব।

## M35-এর সময়কার পুরোনো পরিকল্পনা

M1–M3-এর baseline, M5 body-region experiment, M6 sampled tracking, M7 alignment calculation, M8 dashboard, M9 evaluation tools, M10 sampled-frame playback, M11 dashboard video-run workflow, M12 video annotation pending, M13 withdrawn provisional evaluation, M14 dashboard image/video input, M15 withdrawn annotation prefill, M16 video analysis pause/resume, M17 screenshot presence evaluation, M18 confidence distribution, M19 annotation provenance gate, M20 full-clip coarse tracking, M21 indexed sampling, M22 body-region heatmap response summary, M23 temporary track observation summary, M24 report validation, M25 dashboard summary wrapping, M26 offline-boundary audit এবং M27 planned annotation coverage check, M28 annotation review progress indicator এবং M29 source-frame fingerprint, M30 annotation record serialization test, M31 confidence threshold sweep এবং M32 dashboard threshold-sweep view এবং M33 cross-row consistency gate এবং M34 package/input CLI smoke check এবং M35 threshold-sweep sample composition আছে। পরের কাজ হলো annotation window user-এর desktop-এ সত্যিই দেখা যাচ্ছে কি না নিশ্চিত করা; user প্রস্তুত হলে তবেই তাঁর হাতে নির্বাচিত frame label করা। Verified labels না হওয়া পর্যন্ত video evaluator চালানো হবে না। Body-region evaluator চালাতে visible opponent-এর box-এর ভেতরে head/chest/lower-torso/legs point-ও label করতে হবে। User-reviewed label ও বৈচিত্র্যময় sample না পাওয়া পর্যন্ত detector score final accuracy নয়।

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

### M31: confidence threshold sweep-এর evaluator — কোড আছে, video benchmark নেই

`evaluate_threshold_sweep.py` এক inference pass থেকে বিভিন্ন confidence cutoff-এর metrics গণনার code path দেয়। এর আগের `vd-1` sweep unverified provisional annotation দিয়ে চালানো হয়েছিল; সেই output ও metrics বাতিল এবং মুছে ফেলা হয়েছে। তাই VantaSight-এর `vd-1`-এর threshold comparison এখনো অজানা।

### M32: dashboard-এ threshold sweep দেখানো — UI/loader code আছে

Dashboard threshold-sweep CSV পড়তে ও provenance রাখতে পারে। আগের provisional CSV user-verified ছিল না এবং মুছে ফেলা হয়েছে। কোনো video threshold metric dashboard-এ validated result হিসেবে দেখানো যাবে না যতক্ষণ না user-reviewed labels দিয়ে নতুন report তৈরি হয়।

### M33: threshold-sweep cross-row consistency gate — code behavior

Dashboard loader sweep row-গুলোর scored-frame count, label provenance, IoU threshold ও metric consistency যাচাই করে। এটি report-format validation; detector accuracy বা annotation-এর প্রমাণ নয়।

### M34: package ও input CLI smoke check

Current `.venv`-এ `python -m vantasight` দিয়ে `ss-1.png` load করে `1917×1073` resolution পাওয়া গেছে। একই entry point `vd-1.mp4`-এর জন্য `2560×1440`, `120 FPS`, `7080` frame ও `59s` duration পড়েছে। `python -m pip check`-এ broken dependency পাওয়া যায়নি। এটি metadata/input runtime smoke check; detector accuracy বা dashboard UI test নয়।

### M35: sweep report-এ sample composition — code behavior

Threshold sweep report positive/negative frame ও ground-truth box count বহন করে; dashboard loader row-গুলোর composition consistency পরীক্ষা করে। আগের `vd-1` provisional sample-এর composition ও derived result প্রত্যাহার করা হয়েছে; user-verified sample না হওয়া পর্যন্ত কোনো composition/accuracy ফল রিপোর্ট করা হবে না।

### M36: threshold sweep-এর runtime হিসাব — code behavior

Threshold-sweep CSV-তে model initialization, detector latency, analysis elapsed time ও sampled-frame throughput-এর field আছে; dashboard loader timing metadata-র consistency যাচাই করে। আগের provisional-label run-এর timing record accuracy evidence নয় এবং বর্তমান workflow-তে তা ব্যবহার করা হবে না।

### M37: ছোট target review-এর জন্য annotation zoom ও pan

Annotation window-তে `+`/`-` দিয়ে সর্বোচ্চ `8×` zoom, right mouse drag দিয়ে pan এবং `0` দিয়ে পুরো frame-এ reset যোগ হয়েছে; Windows-এ mouse wheel cursor-এর জায়গায় zoom করে। Existing box/region overlay-ও zoomed view-তে একই source location-এ আঁকা হয়; নতুন box ও body-point আগের মতো source video pixel coordinate-এ save হওয়ার জন্য view crop থেকে coordinate mapping করা হয়েছে। Window-এর নির্দেশনা panel-ও বড় করা হয়েছে, যাতে zoom control-সহ save/quit key দেখা যায়। README-তে নতুন control লেখা হয়েছে। এটি দূরের/ছোট opponent frame খুঁটিয়ে দেখার সুবিধা; auto-enemy classification নয়। Python compile ও CLI help check pass করেছে; live window-তে zoom/pan/box mapping-এর visual check এখনো বাকি।

### M38: runtime metadata consistency — code behavior

Threshold-sweep dashboard loader sampled frame-এর সংখ্যা, analysis time ও inference timing-এর সামঞ্জস্য যাচাই করে। আগের `vd-1` provisional report-এ এই validator চালানো হয়েছিল, কিন্তু user label unverified থাকায় report/result মুছে ফেলা হয়েছে। Validator-এর উপস্থিতি accuracy benchmark নয়.

### M39: annotation window-এর idle CPU ব্যবহার কমানো

একই স্থির frame প্রতি 25 ms-এর বদলে 100 ms event-poll interval ব্যবহার করা হয়েছে। তবে নতুন process idle-এ frame redraw চালিয়ে যাওয়ায় প্রায় 88 wall-second-এ 34 CPU-second ব্যবহার করেছে; polling interval একা যথেষ্ট ছিল না। এ জন্য M40-এ dirty-state redraw যোগ করা হয়েছে।

### M40: annotation frame শুধু পরিবর্তনে redraw

Annotation viewer প্রথমে frame render করে; zoom/pan, mouse drag, label বা keyboard control বদলালে নতুন করে render করে। Idle অবস্থায় `waitKey(100)` শুধু mouse/keyboard event poll করে, full-resolution frame resize ও overlay redraw করে না। নতুন launcher-started window-তে input ছাড়া 15-second sample-এ process CPU time `0.09 s` বেড়েছে (এক logical core-এর প্রায় `0.6%`); এটি দীর্ঘ benchmark নয়। Compile/help check pass করেছে; zoom/box coordinate mapping-এর visual review বাকি।

### M41: Windows-এ এক ক্লিকে annotation চালু

`tools/start_vd1_annotation.bat` repository root-এ গিয়ে `.venv` ও `vd-1.mp4` আছে কি না দেখে, তারপর 12-frame review-কে `vd-1_user_verified.jsonl`-এ চালায়। আগের সংস্করণে unverified provisional box prefill হতো; M66-এ সেটি বাদ দেওয়া হয়েছে। Missing setup/input হলে console-এ error দেখিয়ে থামে; review tool শেষ হলে console খোলা থাকে, যাতে error/summary দেখা যায়। Launcher-এর process check কখনোই annotation window user দেখেছেন—এমন প্রমাণ ছিল না; user নিশ্চিত করেছেন window দেখা যায়নি। GUI window user-এর desktop-এ দৃশ্যমান কি না এখনো verify করা হয়নি।

### M42: IoU-ভিত্তিক 101-point AP — evaluator code আছে, valid video score নেই

`evaluate_iou_ap.py` confidence-ranked prediction ও ground-truth box থেকে IoU `0.50`–`0.95`-এ interpolated AP হিসাব করতে পারে। আগের `vd-1` AP report unverified provisional label-এর ওপর নির্ভর করেছিল; সেসব সংখ্যাকে প্রত্যাহার করে report মুছে ফেলা হয়েছে। তাই `vd-1`-এর AP বর্তমানে অজানা।

### M43: AP report-এর model/run metadata — code behavior

AP CSV model architecture/weights, confidence floor, resize mode, device ও matching method-এর metadata রাখতে পারে; dashboard loader metadata completeness ও consistency যাচাই করে। পুরোনো `vd-1` provisional AP score মুছে ফেলা হয়েছে; metadata validation-কে accuracy evidence হিসেবে ধরা যাবে না.

### M44: dashboard-এ IoU/AP curve

AP report dashboard-এ load হলে AP@IoU point-গুলো একটি responsive Tkinter chart-এ আঁকা হয়। IoU threshold x-axis-এ, AP 0–1 y-axis-এ; provenance, sample size, mean এবং model/run metadata chart-এর উপরের text summary-তে থাকে। Compile ও dashboard CLI help check pass করেছে; window-তে graph-এর visual review বাকি।

### M45: annotation completion status পুনরায় যাচাই

User নিশ্চিত করেছেন annotation window তিনি দেখেননি। কোনো video frame-এর label user confirm করেননি; তাই annotation সম্পন্ন হয়নি। আগের provisional JSONL এবং তার derived reports মুছে ফেলা হয়েছে। Responsive process বা window title থাকলেই user GUI দেখেছেন বা label দিয়েছেন—এটি প্রমাণ হয় না। User পরে annotation করবেন বলেছেন; manual annotation ও video benchmark pending।

### M46: ছোট target-এর জন্য source-resolution inference

`vd-1.mp4` source frame `2560×1440`; আগের default model transform input resize করতে পারত। এখন detector ও experimental pose estimator-এ `--preserve-source-resolution` option আছে; CLI, screenshot/video analysis, tracking, evaluators এবং dashboard local-media panel-এ এটি নির্বাচন করা যায়। Source-resolution mode VRAM বেশি নেয় ও ধীর হতে পারে; accuracy বাড়ার নিশ্চয়তা নেই. Frame 3000-এ candidate inference preview তৈরি হয়েছিল, কিন্তু detections manually annotated/verified নয়। Provisional label-ভিত্তিক AP/threshold তুলনা প্রত্যাহার ও মুছে ফেলা হয়েছে; দুই inference mode-এর accuracy অজানা।

### M47: source-resolution mode-এ user-labeled screenshot pilot

`ss-1` (enemy দৃশ্যমান) ও `ss-2` (enemy দৃশ্যমান নয়)—দুই user-labeled screenshot-এ confidence `0.25`-এ default resize ও source-resolution mode আলাদা করে চালানো হয়েছে। Default resized run: `TP=1, FP=1, TN=0, FN=0`; source-resolution run: `TP=0, FP=1, TN=0, FN=1`। তাই এই run-এ source-resolution mode missed positive screenshot-টি ফেরত আনেনি; negative screenshot-এর false candidate-ও যায়নি। Screenshot `1917×1073`; default transform সামান্য upsample করে প্রায় `1920×1075`, native mode আসল dimension-এই চলে—resize বদলালে confidence cutoff-এর আশেপাশের ফল বদলাতে পারে। মাত্র 2টি screenshot দিয়ে video বা Valorant gameplay accuracy বোঝা যায় না; এই সংখ্যাগুলো mode বেছে নেওয়ার recommendation নয়। দুই run-এর CSV local ignored report; test suite চালানো হয়নি।

### M48: target height অনুযায়ী recall — code আছে, কোনো মাপা ফল নেই

Threshold evaluator ground-truth box height অনুযায়ী tiny/small/medium/large group-এ recall হিসাব করতে পারে। আগের `vd-1`-এর size-specific result unverified provisional box থেকে এসেছিল; সেগুলো বাতিল ও মুছে ফেলা হয়েছে। User-reviewed box ছাড়া target-size recall দাবি করা যাবে না.

### M49: annotation window সামনে ও দৃশ্যমান রাখা

Windows annotation launcher এখন `--keep-on-top` পাঠায়; OpenCV review window অন্য window-এর পেছনে চাপা পড়া ঠেকাতে topmost রাখা হয়। যারা launcher ব্যবহার করছেন না, তারা annotation command-এও এই flag দিতে পারেন। এটি শুধু annotation window-এর UI stacking বদলায়—খেলার screen capture বা game interaction যোগ করে না। Flag ছাড়া annotation tool-এর আগের আচরণ থাকে। Compile ও CLI help check করা হয়েছে; user-এর desktop-এ Explorer থেকে window দেখা যাচ্ছে কি না এখনও হাতে নিশ্চিত করা বাকি, এবং annotation সম্পন্ন হয়েছে বলা যাবে না।

### M50: report-এর source file fingerprint

নতুন screenshot ও video tracking JSON report-এ source file-এর SHA-256 যোগ হয়েছে। Hash streaming 1 MiB block-এ পড়ে; বড় media একবারে memory-তে ওঠে না। Analysis-এর আগে ও পরে hash মেলানো হয়, চলার সময় source বদলালে report লেখা বন্ধ হয়। Dashboard খোলার সময় report hash-এর সঙ্গে নির্বাচিত local source file মেলে; mismatch হলে stale detection overlay ভুল footage-এ দেখানোর বদলে স্পষ্ট error দেয়। Hash-বিহীন পুরোনো report compatibility-র জন্য চলে। File hash video analysis timing-এ যোগ হয় না, তবে analysis start ও dashboard open-এ সম্পূর্ণ file একবার করে পড়া হয়; বড় file-এ অতিরিক্ত I/O সময় লাগতে পারে। Compile এবং পুরোনো/new JSON report loader check pass করেছে; dashboard-এর Windows visual window-তে check দেখা এখনও যাচাই হয়নি।

### M51: evaluation CSV-তেও source fingerprint

Threshold-sweep ও IoU/AP evaluator source video fingerprint CSV-তে লেখে; dashboard tracking report-тай fingerprint মেলাতে পারে. আগের provisional sweep/AP report-গুলো unverified ছিল এবং local data থেকে সরানো হয়েছে। Fingerprint কেবল source file matching নিশ্চিত করে—label correctness বা accuracy নয়.

### M52: সাধারণ video frame preview-এর output option ঠিক করা

README-তে দেখানো `--frame-index ... --output ...` command বাস্তবে চলছিল না: CLI ভুলভাবে `--detect-persons` বাধ্যতামূলক করছিল, যদিও plain frame preview-র জন্য detector লাগে না। Validation বদলে এখন video-র নির্দিষ্ট frame-এর সাধারণ preview-তে custom `--output` চলে; pose-only analysis-এ ভুল করে `--output` দিলে `--pose-output` ব্যবহারের নির্দেশ দেয়। `.venv` দিয়ে `vd-1.mp4` frame `3000` extract করে custom ও default path-এ PNG পাওয়া গেছে; দুই ফাইলের SHA-256 একই (`E307893F…8B004CF`)। Metadata: `2560×1440`, `120 FPS`, 7080 frame, duration 59s; frame 3000 timestamp 25.000s। PNG-টি সরাসরি দেখে নিশ্চিত হয়েছি যে সম্পূর্ণ captured frame, streamer inset-সহ, preview-তে আছে। Preview ও source video local ignored data—Git commit-এ যোগ হয়নি। Manual annotation এই ধাপের অংশ নয় এবং এখনো করা হয়নি।

### M53: Windows-এ local dashboard launcher

`tools/start_vd1_dashboard.bat` যোগ হয়েছে। এটি `.venv`, local `vd-1.mp4` ও tracking JSON আছে কি না দেখে, তারপর offline Tk dashboard চালায়। Fingerprint-সহ threshold-sweep/AP CSV local-এ থাকলে সেগুলোও attach করে; না থাকলেও tracking dashboard খোলে। প্রয়োজনীয় tracking JSON না থাকলে README-র exact generation command console-এ দেখায়। README-তে usage লেখা হয়েছে। Batch file ও argument path staticভাবে যাচাই করা হয়েছে; Windows desktop lock screen-এ থাকায় launcher চালিয়ে dashboard window দেখা সম্ভব হয়নি। UI visual QA pending; annotation চালানো হয়নি।

### M54: dashboard-এর জন্য 59-frame tracking report

`vd-1.mp4`-এ প্রতি 120 source frame (1 second) পরপর sample নিয়ে local `vd-1_dashboard_tracks.json` বানানো হয়েছে। Report-এ 59 sampled frame ও 188 COCO person-candidate detection আছে; GPU run throughput `3.464 sampled frame/s`, mean detector latency `136.5 ms`। Source `2560×1440`, `120 FPS`, SHA-256 `3917b7d5…5447c930`; আলাদা file hash report-এর fingerprint-এর সঙ্গে মিলেছে। Dashboard-এর tracking JSON loader report নিয়েছে, আর fingerprint-সহ threshold sweep-এর 15 row ও IoU/AP report-এর 10 row একই source hash-এ load হয়েছে। Sweep/AP label এখনো `unreviewed_exploratory`; candidate-গুলো enemy/team identity নয়। Report-গুলো `reports/`-এ local ignored file। Windows desktop lock থাকায় dashboard-এ 59 frame বাস্তবে দেখা/চালানো যাচাই এখনো বাকি।

### M55: detector candidate preview spot check

Frame `3000` (25s)-এর detection preview বানিয়ে সরাসরি দেখা হয়েছে এবং `vd-1_dashboard_tracks.json`-এর একই frame-এর সঙ্গে মিলেছে: দুটিতেই 3 candidate, একই source-pixel box ও score আছে। একটি box in-game character-এর চারপাশে (`0.963`, `[1247,527,1345,726]`); দুটি বড় box (`0.993`, `[424,940,1096,1398]`; `0.663`, `[144,748,1085,1389]`) lower-screen streamer/foreground UI-র সঙ্গে overlap করে। Video-র webcam exclusion rectangle থাকলেও filter কেবল box center ওই rectangle-এর মধ্যে কি না দেখে; এই দুই box-এর center বাইরে, তাই বাদ যায়নি। এগুলো provisional spot-check observation—manual label বা confirmed false-positive metric নয়। Preview `data/previews/`-এ local ignored output। Region filtering আরও বদলানোর আগে একাধিক frame user review দরকার, যাতে game target ভুল করে বাদ না যায়।

### M56: full-clip coarse tracking continuity review

59 sampled frame ও 188 person-candidate observation থেকে IoU tracker 127 temporary candidate track বানিয়েছে। তার মধ্যে 96টি (75.6%) এক observation-এ শেষ; observations/track median `1`, p95 `4`, maximum `6`; observed-span median `0s`, p95 `3s`, maximum `7s`। Sampling প্রতি 1s, detector confidence cutoff `0.25`, IoU match cutoff `0.30`, এবং সর্বোচ্চ 1 missed update ছিল। এটি candidate-box continuity-এর exploratory measurement; enemy/player identity, true track recall বা gameplay quality নয়। Detector miss ও বড় false candidate দেখা যাওয়ায় max-missed setting এখনই বাড়ানো হয়নি—user-reviewed frame ছাড়া এতে আলাদা target-কে একই track-এ মেলানোর ঝুঁকি আছে। Full report local ignored `reports/vd-1_dashboard_tracks.json`-এ আছে।

### M57: gameplay frame-এ pose region ও alignment spot check

Frame `3000`-এ COCO Keypoint R-CNN 2 person-candidate-কে pose estimate দিয়েছে; detection preview-র মতোই webcam/foreground area-র আরেকটি বড় candidate এসেছে। In-game character-এর candidate box `[1242,533,1348,728]`, detector score `0.996`; head/chest/lower-torso/legs estimate center থেকে frame-center crosshair `(1280,720)`-এর distance যথাক্রমে `158.86`, `148.85`, `103.41`, `31.40` pixel। অন্য candidate-এর বড় box `[377,923,1082,1393]` streamer/foreground overlay-র সঙ্গে মিলে গেছে; তার pose estimate-কে body-region success ধরা যাবে না। Preview `data/previews/vd-1_frame_003000_body_regions.png`-এ সরাসরি দেখা হয়েছে। Model COCO-trained; mean keypoint heatmap response raw score, probability নয়। Distance কেবল pixel geometry—hit probability, target identity বা accuracy নয়। Manual region point label না থাকায় localization quality অজানা; output local ignored preview।

### M58: একই frame-এ resize mode তুলনা

Frame `3000`-এ default TorchVision resizing ও `--preserve-source-resolution`—দুই mode-তেই 3 person-candidate পাওয়া গেছে। Main in-game candidate-এর box IoU `0.9037`; detector score `0.963` থেকে `0.982` হয়েছে, কিন্তু box-count বাড়েনি। Lower-screen foreground/streamer-সংশ্লিষ্ট বড় candidate-ও দুই run-এ রয়ে গেছে (`0.993→0.986` এবং `0.663→0.269`, default cutoff `0.25`)। এটি একটি frame-এর unreviewed output; confidence বদল মানেই accuracy বদল নয়, আর এতে tiny target recall যাচাই হয়নি। Native-resolution mode বেশি GPU memory ও সময় নিতে পারে। দুই preview `data/previews/`-এ local ignored output; accuracy comparison-এর জন্য representative user-reviewed frame দরকার।

### M59: pose model-এর resize বনাম source-resolution comparison

Frame `3000`-এ default pose mode 2 person-candidate estimate দিয়েছে, source-resolution mode 6টি। Native mode-এর অতিরিক্ত 4টি estimate lower-screen streamer/foreground area-তে পড়ে। Main in-game candidate-এর head/chest/lower-torso/legs crosshair distance default mode-এ `158.86/148.85/103.41/31.40 px`; source-resolution mode-এ `160.80/146.33/101.43/32.65 px`—point-ভেদে প্রায় `1–3 px` পার্থক্য। Score `0.996`-ից `0.983` হয়েছে. এটি এক frame-এর spot check; false positive/label ground truth ছাড়া source-resolution pose mode ভালো বা খারাপ—এমন ব্যাপক সিদ্ধান্ত নয়। Default pose resize অপরিবর্তিত; দুই preview local ignored `data/previews/`-এ আছে।

### M60: dashboard runtime launch check

`.venv` থেকে dashboard-টি 59-frame `vd-1` tracking report, fingerprint-সহ threshold sweep ও IoU/AP CSV দিয়ে চালু করা হয়েছে। Windows process status-এ `VantaSight Offline Dashboard — vd-1.mp4` title-সহ responding Tk window দেখা গেছে; startup-এ Python error আসেনি। কিন্তু computer-use window inventory ওই Python window-টি দেখায়নি, তাই dashboard-এর actual layout, playback, frame overlay ও evaluation panel চোখে দেখে যাচাই করা যায়নি। এটি launch check মাত্র—visual QA pass নয়। Video frame-এর manual annotation এখনো deferred; screenshot presence-এর user-labeled yes/no row আলাদা, সেগুলো video box annotation নয়। Dashboard process-টি বন্ধ না করে রাখা হয়েছে; পরের ধাপে window UI inspection, playback/slider interaction, overlay/report consistency check, clipping review, findings fix/documentation. Local report/video CSV-গুলো Git-ignored অবস্থায় আছে।

### M61: screenshot presence-এর এক-pass threshold sweep

`evaluate_screenshot_presence.py`-তে `--thresholds` যোগ হয়েছে। একবার প্রতি screenshot-এ inference করে প্রস্তাবিত confidence cutoff-গুলোতে candidate presence ও TP/FP/TN/FN/precision/recall-সহ summary হিসাব করে; CSV-তে প্রতি screenshot ও cutoff-এর row এবং source image-এর SHA-256 থাকে। `ss-1` (enemy visible) ও `ss-2` (enemy not visible)-এর user-provided label-এ default resized inference, CUDA, max-side `1920`-এ 6টি cutoff চালিয়েছি। Confidence `.25`-এ `TP=1, FP=1`; `.26` ও `.28`-এ দুটো ছবিই সঠিক binary presence দিয়েছে; `.30` থেকে positive `ss-1` miss হয়েছে। Recorded candidate score `.255909` (negative `ss-2`) এবং `.286332` (positive `ss-1`), তাই এই দুই ছবিতে আলাদা করার threshold interval `(0.255909, 0.286332]`; এটি মাত্র দুইটি sample-এর narrow separation, general calibration বা gameplay performance evidence নয়। 12-row fingerprint-সহ CSV `reports/screenshot_presence_threshold_sweep.csv` local ignored output। README-তে ব্যবহার ও সীমাবদ্ধতা যোগ হয়েছে; video evaluator ও ground-truth label অপরিবর্তিত।

### M62: dashboard playback layout-এর সংশোধন

Dashboard window সামনে তুলে আগের visual inspection-এ report metadata, AP curve, confidence sweep, sampled gameplay frame ও local-media panel দেখা গেছে। একই দৃশ্যে Previous/Next/Play/Pause/Stop control viewport-এর নিচে কেটে যাচ্ছিল। `open_dashboard.py`-তে controls ও playback note-কে image panel-এর bottom-এ pack করে canvas-কে অবশিষ্ট জায়গায় expand করানো হয়েছে; Tk window startup-এ normal/deiconified এবং raise/focus করার নির্দেশও যোগ হয়েছে। `git diff --check` pass করেছে। এই code change-এর পর নতুন launch করা window computer-use window inventory-তে উপস্থিত হয়নি; visible পুরোনো window-তে controls এখনও clipped, তাই নতুন layout-এর visual verification অসম্পূর্ণ—এটিকে UI pass বলা হচ্ছে না। পরের ধাপে desktop-এ updated window দেখা গেলে control visibility এবং playback আচরণ যাচাই করতে হবে। Video manual annotation এখনও deferred; আগের annotation completed বলে কোনো verified label নেই, `vd-1_user_verified.jsonl`-ও নেই। Dashboard-এর candidate/AP/threshold result-গুলো unreviewed exploratory; user-verified gameplay ground truth নয়।

### M63: dashboard থেকে custom crosshair reference

Local-media analysis panel-এ custom crosshair `x,y` দেওয়ার field যোগ হয়েছে। মান source image-এর top-left থেকে pixel coordinate; field ফাঁকা রাখলে আগের মতো frame-center reference ব্যবহৃত হবে। Screenshot analyzer `--crosshair x,y` নিয়েও কাজ করে; এটি `--measure-alignment` ছাড়া দেওয়া যাবে না, এবং coordinate নির্বাচিত frame-এর সীমানার বাইরে হলে analysis শুরু হয় না। Dashboard-এ একই validation আছে এবং screenshot/video—দুই runner-এই coordinate পাঠায়। এতে অন্য FPS-এর footage-এ reticle যদি frame-এর geometric center-এ না থাকে, offline geometry reference সেট করা যাবে। এটি শুধু pixel distance; target identity, hit probability বা aim accuracy নয় এবং game input পাঠায় না। README ও CLI usage update হয়েছে। Syntax ও CLI help যাচাই হয়েছে; dashboard visual verification এখনো বাকি; manual video annotation deferred ও verified label অনুপস্থিত।

M63 যাচাই: সংশ্লিষ্ট তিনটি Python file `compileall`-এ compile হয়েছে; screenshot analyzer ও video tracker-এর CLI help-এ `--crosshair` দেখা গেছে। `ss-1.png`-এ CUDA-তে প্রকৃত offline analysis চালিয়ে JSON report পরীক্ষা করেছি: 1917×1073 source, 1 person candidate, `settings.crosshair_xy=[100.0,100.0]`, এবং region distance `942.95px` / normalized `0.429225` ওই reference থেকে হিসাব হয়েছে। এটি option-to-report plumbing-এর যাচাই, detector বা pose accuracy-এর প্রমাণ নয়। Dashboard form-এ coordinate বসিয়ে চালানোর visual check CUA window access না থাকায় এখনো বাকি; video manual annotation deferred-ই আছে।

ভিডিও runner-ও যাচাই: `vd-1.mp4` frame `3000`-এ এক-frame source-resolution-preserving নয় এমন default offline run-এ 3 candidate এসেছে; JSON-এ custom crosshair `[100.0,100.0]` এবং 8টি estimated-region distance ওই reference থেকে লেখা হয়েছে। এটি শুধু video CLI path-এর wiring; 1 frame-এর ফল accuracy evidence নয়।

### M64: dense source-resolution dashboard report

`tools/start_vd1_dashboard.bat` এখন `reports/vd-1_dashboard_tracks_dense.json` খোলে; report না থাকলে `.venv` command-এ `--frame-step 30 --preserve-source-resolution` দেখায়। README-তে একই generation command যোগ হয়েছে। `vd-1.mp4`-এ default confidence `0.25`, IoU `0.30`, `cuda:0`, source resolution `2560×1440` রেখে 236 sampled frame (`0` থেকে `7050`, প্রতি 30 source frame) চালিয়ে 816 COCO person-candidate detection পাওয়া গেছে; 27 sampled frame-এ candidate ছিল না। Analysis throughput `1.351 sampled frame/s`, detector latency mean `255.0ms`, p95 `287.6ms`; 386 temporary box tracks-এর মধ্যে 256 single-observation track, observed-span p95 `1.75s`, max `7.0s`। এগুলো sampling-এ পাওয়া generic person candidate—27টি frame-এ enemy ছিল না বা কোনো track একই player—এমন প্রমাণ নয়। পুরোনো 1-second report-এর সঙ্গে count/metric সরাসরি তুলনীয় নয়, কারণ sampling density ও inference resolution আলাদা। Dense report `reports/`-এ local ignored; GitHub-এ code/docs-ই push হবে। Dashboard-এর visual playback verification ও user-verified manual video annotation এখনো বাকি।

### M65: candidate box height summary

`tools/summarize_candidate_sizes.py` validated tracking JSON নেয় এবং sampled frame-পিছু CSV row লেখে। Box height-কে source frame height-এর শতাংশ ধরে `tiny (<2%)`, `small (2–6%)`, `medium (6–16%)`, `large (≥16%)` group করা হয়; zero-candidate frame-ও CSV-তে থাকে। এটি predicted person-candidate box-গুলোর আকার দেখায়, enemy identity, true target size বা recall নয়। `vd-1_dashboard_tracks_dense.json`-এ চালালে 236 row/816 candidate পাওয়া গেছে: tiny `0`, small `36`, medium `210`, large `570`; 27 frame-এ candidate ছিল না। এই 36 small box manual review-এর জন্য খুঁজে দেখা যেতে পারে, কিন্তু সেগুলো enemy বা সঠিক detection—এমন প্রমাণ নেই। Script `compileall` ও CLI help-এ যাচাই, এবং dense local report-এ actual CSV লেখা হয়েছে। CSV `reports/`-এ local ignored; manual video annotation ও dashboard visual QA এখনো বাকি।

### M66: unverified labels থেকে annotation ও dashboard আলাদা রাখা

আগের Windows annotation launcher provisional JSONL থেকে box suggestion দেখাত; README-তেও সেই নির্দেশ ছিল। User জানান annotation window তিনি দেখেননি এবং কোনো label confirm করেননি। তাই annotation launcher ও README-র `vd-1` command থেকে `--prefill` সরানো হয়েছে—পরে annotation চালালে frame scratch থেকে শুরু হবে এবং output `vd-1_user_verified.jsonl`-এ যাবে। Dashboard launcher-এও পুরোনো provisional threshold-sweep ও IoU/AP CSV স্বয়ংক্রিয়ভাবে লোড করার ব্যবস্থা বাদ দেওয়া হয়েছে; এটি এখন শুধু dense tracking report খোলে। এই রিপোর্টে generic model candidate আছে, verified enemy detection বা accuracy নয়। এখন কোনো annotation GUI launch বা annotation করা হয়নি। User annotation পরে করবেন বলেছেন; annotation window visible হওয়া desktop-এ এখনো verify করা বাকি।

### M67: dashboard-এর দৃশ্যমান layout আংশিক পর্যালোচনা

Computer-use-এ existing `VantaSight Offline Dashboard — vd-1.mp4` window foreground-এ এনে screenshot review করেছি। Header-এ source metadata, candidate count, temporary-track summary ও “Offline COCO person candidates only” warning দেখা যায়। Previous/Next/Play samples/Pause/Stop buttons, local-media panel এবং custom `Crosshair x,y` field-ও দৃশ্যমান; screenshot-এ এগুলো clipped নয়। কিন্তু live window-এর report label ছিল `reports/vd-1_dashboard_tracks.json`, dense report নয়; AP ও threshold panels-এ `unreviewed_exploratory` লেখা ছিল। এগুলো invalidated provisional label থেকে আসা, তাই ফল হিসেবে ব্যবহার করা যাবে না। Next/Play click-এর পর frame বদলেছে—এমন নির্ভরযোগ্য visual confirmation পাইনি। সংশোধিত launcher-এ যে dense report খুলবে তা এখনও desktop-এ verify করা হয়নি; playback interaction-ও pending।

### M68: dense report ও source video-র fingerprint মিল

PowerShell দিয়ে `reports/vd-1_dashboard_tracks_dense.json`-এর `source.source_sha256` এবং বর্তমান `data/input/videos/vd-1.mp4`-এর SHA-256 তুলনা করেছি। দুটোই `3917b7d52fe197c69d9305a698a50ff087e9378f01b3ab8c3f26668e5447c930`; dense report-এ 236 sampled frame ও 816 candidate লেখা আছে। এতে dashboard loader-এর source-fingerprint gate পাস করার পূর্বশর্ত মেলে। এই integrity check report-এর সঠিক source file নিশ্চিত করে; dashboard window launch, playback, enemy identity বা detector accuracy নিশ্চিত করে না। Report ও video Git-ignored local files।

### M69: local dashboard shortcut সংশোধন

Ignored `reports/VantaSight Dashboard Review.lnk` COM properties পড়ে দেখা যায় এটি `m50_tracks.json` এবং provisional threshold/AP CSV-তে point করছিল। Shortcut-এর target `tools/start_vd1_dashboard.bat`, arguments ফাঁকা এবং working directory repository root করা হয়েছে। আবার shortcut পড়ে তিনটি value verify করেছি। Batch launcher Git-tracked এবং dense report খোলে; `.lnk` local-only, তাই GitHub-এ commit হয়নি। Shortcut double-click করে নতুন dashboard window targetable কি না নিশ্চিত করতে পারিনি; launch attempt-এর background process বন্ধ করেছি।

### M70: annotation launcher-এর startup নির্দেশনা

`tools/start_vd1_annotation.bat` এখন `vd-1` input ও virtual environment যাচাইয়ের পর জানায় যে review ফাঁকা থেকে শুরু হবে, provisional box আসবে না, এবং expected window title prefix কী। Error হলে console বন্ধ না করে পড়তে বলে। README-তেও এই আচরণ লেখা হয়েছে। এতে GUI সত্যিই খুলেছে—এমন দাবি নয়; user annotation পরে করবেন, তাই এই ধাপে tool launch বা label save করা হয়নি।

### M71: dashboard window-তে report name

`tools/open_dashboard.py`-র window title-এ source filename-এর পরে report filename stem যোগ হয়েছে। ফলে একই `vd-1.mp4`-এর পুরোনো coarse report এবং `vd-1_dashboard_tracks_dense.json`-এর window একই নামে গুলিয়ে যাবে না। README-তে title-এর নিয়ম লেখা হয়েছে। এটি window-identification change; dashboard launch বা playback interaction এখনো আলাদাভাবে verify করা দরকার।

### M72: নতুন dashboard launch attempt-এর ফল

সর্বশেষ source ও matching dense report দিয়ে `.venv\Scripts\python.exe tools\open_dashboard.py reports\vd-1_dashboard_tracks_dense.json` চালানো হয়েছিল। Process চলমান ছিল, কিন্তু CUA window inventory-তে নতুন report-specific title আসেনি এবং screenshot তোলা যায়নি; terminal-এও error text আসেনি। কেবল ওই launch attempt-এর console session বন্ধ করেছি। এটি সফল visible launch নয়। পরের UI verification-এ user-এর desktop-এ shortcut double-click করে দেখা বা visible console-এর error/output সংগ্রহ করা দরকার।

### M73: unverified annotation ও metric পরিষ্কার

User নিশ্চিত করেছেন যে annotation window তিনি দেখেননি। তাই `vd-1`-এর provisional JSONL-কে annotation হিসেবে গণ্য করা ভুল; user-confirmed label বা metric নেই। Git-ignored `data/annotations/vd-1_provisional_review.jsonl` এবং তার সঙ্গে মিলে থাকা 10টি provisional/recheck CSV report local machine থেকে সরানো হয়েছে। `vd-1_frame_review.csv` ও `vd-1_threshold_comparison.csv`-এর unverified classification-ও মুছে ফেলা হয়েছে। Workflow-র পুরোনো threshold table, provisional score এবং hand-review classification দাবিগুলো প্রত্যাহার করে বর্তমান অবস্থাকে পরিষ্কার করা হয়েছে। এই পরিবর্তন কেবল unverified artifact/দাবি মুছে; কোনো নতুন annotation, detector evaluation বা accuracy result তৈরি করেনি। `git diff --check` দিয়ে whitespace যাচাই করে তারপর docs-only change commit/push করা হবে.

### M74: dashboard যেন নিজে থেকে evaluation CSV না তোলে

চালু থাকা পুরোনো dashboard window-তে coarse `vd-1` report-এর পাশে `unreviewed_exploratory` AP ও threshold result দেখা গেছে। এগুলো আগের unverified label থেকে তৈরি; accuracy হিসেবে ব্যবহার করা যাবে না। Code audit-এ পাওয়া গেছে যে dashboard command-এ evaluation path না দিলেও `reports/detector_evaluation.csv` ও `reports/pose_region_evaluation.csv` নামে local file থাকলে সেগুলো auto-load হতো। `tools/open_dashboard.py`-তে এই default loading সরিয়ে এখন detector/pose CSV কেবল explicit `--detector-evaluation` / `--pose-evaluation` path দিলে load হয়। README ও M8-ও সেই নিয়ম জানায়; dense dashboard batch launcher কোনো evaluation CSV path দেয় না।

বর্তমান খোলা window পুরোনো process/report; এই code change সেটির in-memory display বদলায় না। Computer-use refresh-এর সময় window-তে user input শনাক্ত হওয়ায় আর UI action করিনি। ফলে updated dense report-এর visible window ও playback এখনও verify হয়নি। `git diff --check` clean; test চালানো হয়নি।

### M75: dense dashboard process চালু, visual review অসম্পূর্ণ

Code update-এর পর `.venv\Scripts\python.exe tools\open_dashboard.py reports\vd-1_dashboard_tracks_dense.json` সরাসরি চালিয়েছি। Process live আছে এবং Windows-এর window enumeration-এ নতুন window title `VantaSight Offline Dashboard — vd-1.mp4 — vd-1_dashboard_tracks_dense` পাওয়া গেছে। কিন্তু Computer Use-এর `list_windows` এখনও শুধু পুরোনো `VantaSight Offline Dashboard — vd-1.mp4` window দেখাচ্ছে; নতুন window-তে screenshot/input দেওয়া যাচ্ছে না। তাই title/process দেখা গেছে বলা যায়, dense report-এর content, no-metric default, layout বা playback visually verified বলা যায় না। পুরোনো window-তে এখনো আগের in-memory `unreviewed_exploratory` AP/threshold panel দেখা যাচ্ছে; নতুন code সেই চলমান পুরোনো process-কে বদলায় না। নতুন process বন্ধ করিনি; একই live process handle রেখে next window refresh-এ পুনরায় দেখতে হবে, duplicate launch করা যাবে না।

### M76: GitHub-এ সীমিত demo preview যোগ

`ss-1`, `ss-2` ও `vd-1` frame 3000-এর detector preview থেকে তিনটি JPEG বানিয়ে `docs/assets/`-এ রাখা হয়েছে; মোট আকার প্রায় 0.9 MB। `vd-1` copy-তে per-source webcam rectangle কালো করে ঢেকে দেওয়া হয়েছে, কিন্তু model-এর আগে থেকে আঁকা green candidate box-গুলো অপরিবর্তিত আছে—তাই webcam-সংলগ্ন box-ও verified target নয়। README-তে পাশাপাশি screenshot-গুলো এবং video-frame preview দেখানো হয়েছে, আর caption-এ in-game character candidate ও `ss-2` arm/weapon false-positive-র উদাহরণ ব্যাখ্যা করা হয়েছে। সব preview model output; manual annotation বা detector accuracy-র প্রমাণ নয়। `git diff --check` ও image content/size review করা হয়েছে; test চালানো হয়নি। Raw video বা local annotation/evaluation data যোগ করা হয়নি।

### M77: dashboard verification-এর বর্তমান machine state

এই ধাপের শুরুতে PowerShell process inventory-তে কোনো `python`/`pythonw` process পাওয়া যায়নি, তাই M75-এ চালানো dense-report window/process এখন আর live নয়। Computer Use-এর `list_windows` native pipe connection error দিয়ে ব্যর্থ হয়েছে; একবার retry এবং kernel reset করে পুনরায় চালানোর পরও একই error এসেছে। UI-তে click/input বা নতুন launch করা হয়নি। আগের stale dashboard screenshot-এ unverified AP/threshold ফল ছিল; সেগুলো ব্যবহারযোগ্য নয়। পরের dashboard attempt-এর আগে Computer Use pipe সচল হতে হবে, তারপর `tools/start_vd1_dashboard.bat` দিয়ে নতুন dense-report window খুলে তার title/report label, no-evaluation panel, sampled-frame controls ও Next/Play আচরণ চোখে দেখে যাচাই করতে হবে। বর্তমান machine-এ dashboard UI QA pending; video annotation-ও user পরে করবেন বলেছেন।

### M78: user review-এর জন্য deterministic frame shortlist

`tools/create_review_shortlist.py` validated tracking report-এর sampled frame-গুলো ছয়টি chronological bin-এ ভাগ করে। প্রতি bin-এ সবচেয়ে ছোট person-candidate box থাকা একটি frame এবং থাকলে শূন্য person-candidate-সহ একটি frame বেছে নেয়। `vd-1_dashboard_tracks_dense.json`-এ চালিয়ে 12টি frame পেয়েছি: 6টি ছোট candidate-সহ ও 6টি zero-candidate। Output `reports/vd-1_manual_review_shortlist.csv`-এ frame index, timestamp, candidate count, source fingerprint ও selection reason আছে; এটি Git-ignored local review aid। Small box মানে enemy নয়, আর zero-candidate মানে enemy নেই—এমন নয়। কোনো label তৈরি বা evaluator চালানো হয়নি। README-তে command যোগ হয়েছে। Report-এ actual run দিয়ে যাচাই করেছি; test suite চালানো হয়নি। পরের review-তে CSV-এর frame index user নিজে দেখে annotation window দৃশ্যমান হলে তবেই label করবেন।

### M79: shortlist-কে annotation-এর planned sample করা

`vd-1_manual_review_shortlist.csv`-এর বর্তমান 12টি frame index `540,1080,1650,1800,3030,3210,3840,4650,4890,4980,6090,6480` annotation command ও Windows launcher-এ বসানো হয়েছে। এতে প্রতিটি chronological bin-এ model-এর সবচেয়ে ছোট candidate frame এবং candidate শূন্য থাকা frame—দুটিই review sample-এ আসে, তবে frame selection model output-ভিত্তিক; এটি label balance বা enemy presence নিশ্চিত করে না। Video বদলালে নতুন tracking report থেকে shortlist আবার তৈরি করে command/launcher-এর index হালনাগাদ করতে হবে। Threshold-sweep ও IoU/AP command-এর `--expected-frames`-ও একই sample list-এ মেলানো হয়েছে, যাতে পরে user-confirmed annotation করলে coverage gate সামঞ্জস্যপূর্ণ থাকে। কোনো annotation GUI চালানো বা label লেখা হয়নি।

এই ধাপে Computer Use-এর `list_apps` আবার native pipe connection error দিয়েছে, তাই dashboard বা annotation window-এর দৃশ্যমানতা যাচাই হয়নি। User annotation পরে করবেন বলেছেন; launcher-এর update কেবল প্রস্তুতি, annotation সম্পন্ন হওয়ার প্রমাণ নয়। `ss-1`/`ss-2` presence evaluator চালিয়ে M17-এর pilot result (`TP=1, FP=1` @0.25) পুনরুৎপন্ন হয়েছে; sample n=2 হওয়ায় এটি নতুন general performance claim নয়। Test suite চালানো হয়নি।

### M80: shortlist CSV থেকে সরাসরি annotation frame নেওয়া

`tools/annotate_video_frames.py`-তে `--shortlist-csv` যোগ হয়েছে। Tool CSV থেকে unique, non-negative frame index নেয় এবং annotation window খোলার আগে `source_video` নাম ও source video-র SHA-256 মিলিয়ে দেখে; অন্য video বা বদলে যাওয়া source-এর shortlist reject হয়। পুরোনো `--frames` command-ও রাখা হয়েছে; দুটির মধ্যে একটিই দিতে হবে। `tools/start_vd1_annotation.bat` `reports/vd-1_manual_review_shortlist.csv`-কে input করে, file না থাকলে dense report-থেকে shortlist বানানোর command দেখায়। README ও workflow-তে CSV-ভিত্তিক command update হয়েছে। বর্তমান `vd-1.mp4` ও shortlist CSV দিয়ে source-fingerprint validation চালিয়ে প্রত্যাশিত 12টি index পেয়েছি; CLI help-এ mutually exclusive input দুটিও দেখা গেছে। GUI runtime চালানো হয়নি, কোনো annotation file লেখা হয়নি, test suite চালানো হয়নি।

### M81: annotation launcher-এ shortlist auto-generation

Windows launcher আগে CSV না পেলে শুধু তৈরির নির্দেশ দিত। এখন dense tracking report পাওয়া গেলে launch-এর সময় `create_review_shortlist.py` চালিয়ে স্থানীয় CSV তৈরি করে; dense report না থাকলে প্রয়োজনীয় tracking command console-এ দেখিয়ে থামে। এরপর annotation tool CSV-এর source fingerprint validate করে। README-তে নতুন আচরণ লেখা হয়েছে। Batch launcher run বা GUI open করা হয়নি; test suite চালানো হয়নি।

### M82: annotation-এর আগে frame preflight

Annotation tool-এ `--dry-run` যোগ হয়েছে: shortlist fingerprint যাচাই করে নির্বাচিত frame-গুলো video থেকে decode করে, কিন্তু GUI খোলে না এবং annotation file লেখে না। Windows launcher-ও GUI খোলার আগে এই preflight চালায়। `vd-1.mp4` (2560×1440)-এর বর্তমান shortlist-এ 12/12 frame decode হয়েছে: 540, 1080, 1650, 1800, 3030, 3210, 3840, 4650, 4890, 4980, 6090, 6480। Command শেষে স্পষ্ট জানিয়েছে window খোলা বা label লেখা হয়নি; output file-ও নেই। `git diff --check` পরিষ্কার; test suite চালানো হয়নি।

### M83: sampled report-এ candidate-শূন্য gap summary

`summarize_zero_candidate_runs()` sampled frame-এ model-এর detection শূন্য থাকার সংখ্যা, অনুপাত, run-সংখ্যা এবং দীর্ঘতম adjacent run-এর frame/timestamp হিসাব করে। Dashboard-এ এই তথ্য person-candidate output হিসেবে দেখানো হয় এবং warning-এ বলা হয়, candidate না পাওয়া enemy অনুপস্থিতির প্রমাণ নয়। Dense `vd-1` report-এর 236 sample-এ 27টি candidate-শূন্য sample (11.4%), 15টি আলাদা run; দীর্ঘতম run 10 sample, frame 3570–3840 (29.75–32.0s)। এটি detector miss, inference cadence বা game scene—কোন কারণে হয়েছে তা একা নির্ধারণ করে না। বর্তমান dense JSON দিয়ে summary function চালিয়ে ফল পেয়েছি; dashboard GUI-তে নতুন লাইন দেখা যাচাই হয়নি, test suite চালানো হয়নি।

### M84: zero-candidate frame-এর detail সতর্কতা

Dashboard-এ কোনো sampled frame-এ detection না থাকলে detail panel এখন জানায় যে model person candidate ফেরায়নি এবং এটিকে enemy অনুপস্থিতির প্রমাণ ধরা যাবে না। README-তেও individual-frame caveat যোগ হয়েছে। `git diff --check` clean; dashboard GUI-তে text wrap বা দৃশ্যমানতা যাচাই হয়নি, test suite-ও চালানো হয়নি।

### M85: reproducible sample media GitHub-এ

দুটি source screenshot, `screenshot_presence.csv`-এর user-provided yes/no label, এবং HUD exclusion-region configuration repository-তে যোগ হয়েছে। GitHub clone-এ default screenshot evaluator তাই original sample input পাবে। 212,593,238-byte `vd-1.mp4`-কে `.gitattributes`-এ `*.mp4` Git LFS দিয়ে track করেছি; commit `ed8c3cf` push হয়েছে এবং `git lfs push origin main` exit code 0 দিয়েছে। M85-এ fresh clone থেকে fetch তখন আটকে গিয়েছিল; M95-এ পরে সফলভাবে যাচাই হয়েছে। নতুন video annotation file commit হয়নি।

### M86: manual annotation window খোলা

User annotation window দেখতে পাননি বলে আগের `--dry-run`-কে annotation হিসেবে ধরা হয়নি; এটি শুধু frame preflight ছিল। এবার project `.venv`-এর OpenCV WIN32 GUI দিয়ে `tools/annotate_video_frames.py` চালিয়ে actual window খুলেছি। Window title `VantaSight annotation - vd-1.mp4 frame 540`; ১২-frame shortlist-এর প্রথম frame দেখানো হচ্ছে। Annotation output file এখনো তৈরি হয়নি, কোনো frame label/box-ও save হয়নি। User frame নিজে দেখে `y` (opponent দেখা যায়), `n` (দেখা যায় না), অথবা `u` (অনিশ্চিত) বেছে নেবেন; `Enter` দিয়ে save/পরের frame, `q` দিয়ে বাকি review থামানো যায়। Box আঁকতে frame-এর ওপর left-drag করতে হয়। User review শেষ হওয়ার আগে কোনো video benchmark নেই।

### M87: screenshot presence pilot পুনরায় চালানো

User-labeled `ss-1`/`ss-2` দিয়ে current detector-এ দুইটি source mode তুলনা করেছি (`cuda:0`, দুইটি image, resolution 1917×1073 ও 1917×1070)। Default resize mode-এ threshold 0.25-এ TP=1, FP=1, TN=0, FN=0; `ss-1` doorway character score 0.286, `ss-2` first-person arm/weapon false candidate score 0.256। `--preserve-source-resolution`-এ একই threshold-এ TP=0, FP=1, TN=0, FN=1; `ss-1`-এর একই box score 0.193-এ নেমেছে, তাই cutoff-এর নিচে গেছে। Source-resolution mode সবসময় detection উন্নত করবে—এমন নিশ্চয়তা নেই। ওই mode-এর threshold sweep-এ `ss-1` max score 0.1934 এবং `ss-2` false candidate score 0.3650; 0.05–0.15-এ positive screenshot ধরা পড়লেও false positive-টিও থাকে। দুইটি screenshot দিয়ে general detector accuracy বা bounding-box localization প্রমাণ হয় না; এগুলো small-target/false-positive ঝুঁকির pilot উদাহরণ। Local CSV `reports/screenshot_presence_eval_default_latest.csv`, `reports/screenshot_presence_eval_latest.csv` ও `reports/screenshot_presence_threshold_sweep_latest.csv`; এগুলো generated report হওয়ায় Git-ignored।

পুনরায় চালাতে:

```powershell
.venv\Scripts\python.exe tools\evaluate_screenshot_presence.py --confidence-threshold 0.25 --preserve-source-resolution --output reports\screenshot_presence_eval_latest.csv
.venv\Scripts\python.exe tools\evaluate_screenshot_presence.py --confidence-threshold 0.25 --output reports\screenshot_presence_eval_default_latest.csv
.venv\Scripts\python.exe tools\evaluate_screenshot_presence.py --thresholds 0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70,0.80,0.90,0.95 --preserve-source-resolution --output reports\screenshot_presence_threshold_sweep_latest.csv
```

### M88: screenshot pair-এ pose association probe

`ss-1` ও `ss-2`-তে detector-এর পর একই COCO Keypoint R-CNN settings দিয়ে exploratory pose analysis চালিয়েছি। `ss-1`-এর 31×49 px doorway candidate pose box-এ match হয়েছে (`pose_person_score=0.966`), চারটি coarse body-region estimate ও source-pixel preview marker-ও তৈরি হয়েছে। `ss-2`-এর 185×340 px first-person arm/weapon candidate `pose_status=unmatched`; কোনো pose box তার সঙ্গে match হয়নি। এই দুই উদাহরণে pose association ছোট target রেখে arm false positive আলাদা করেছে বলে দেখায়, কিন্তু নমুনা অত্যন্ত ছোট এবং pose/person score বা raw heatmap response calibrated confidence নয়। তাই detector-কে pose দিয়ে auto-filter করা হয়নি; video frames-এ manual label দিয়ে পরীক্ষা বাকি। Local output `reports/ss-1_pose_alignment_latest.json`, `reports/ss-2_pose_alignment_latest.json` ও `data/previews/ss-1_body_regions.png`।

পুনরায় চালাতে:

```powershell
.venv\Scripts\python.exe tools\analyze_image.py "data\input\screenshots\ss-1.png" --estimate-body-regions --measure-alignment --output reports\ss-1_pose_alignment_latest.csv --json-output reports\ss-1_pose_alignment_latest.json
.venv\Scripts\python.exe tools\analyze_image.py "data\input\screenshots\ss-2.png" --estimate-body-regions --measure-alignment --output reports\ss-2_pose_alignment_latest.csv --json-output reports\ss-2_pose_alignment_latest.json
```

### M89: Git LFS remote fetch পুনরায় যাচাই

সেই সময় `git lfs fsck --objects` local object valid জানিয়েছিল এবং `git lfs push origin main` exit code 0 দিয়েছিল, কিন্তু fresh shallow clone fetch তখনও আটকে ছিল; M95-এ নতুন normal shallow clone দিয়ে আবার যাচাই করা হয়েছে। Screenshot dataset GitHub-এ ordinary Git file হিসেবে আছে।

### M90: CPU inference fallback pilot

একই user-labeled দুই screenshot, default resize ও confidence threshold 0.25 দিয়ে `--device cpu` চালিয়েছি। CPU ফল GPU default mode-এর presence prediction-এর সঙ্গে মিলে: `ss-1`-এ candidate (`TP=1`), `ss-2`-এ false candidate (`FP=1`), `TN=0`, `FN=0`। দুই device-এ confidence সামান্য আলাদা ছিল (`ss-1`: CPU 0.2853/GPU 0.2863; `ss-2`: CPU 0.2582/GPU 0.2559), তাই score-গুলো bit-identical বা device-independent নয়। CPU path cache করা weights দিয়ে সফল হয়েছে; এই দুই-screenshot pilot hardware-general performance প্রমাণ করে না। Local output `reports/screenshot_presence_eval_cpu_latest.csv` Git-ignored।

পুনরায় চালাতে:

```powershell
.venv\Scripts\python.exe tools\evaluate_screenshot_presence.py --confidence-threshold 0.25 --device cpu --output reports\screenshot_presence_eval_cpu_latest.csv
```

### M91: browser-এ manual review gallery

OpenCV window না দেখা গেলে user যেন একই fingerprint-checked sample frame হাতে review করতে পারেন, তাই `tools/export_review_gallery.py` যোগ করেছি। এটি video SHA-256 যাচাই করে shortlist-এর ১২টি frame source resolution-এ JPEG-এ export করে এবং প্রতি frame-এর raw-pixel SHA-256-সহ manifest বানায়। `data/previews/vd-1_review_gallery/index.html` একটি standalone offline HTML page: user নিজে opponent status (`y`/`n`/`u`) বেছে নিতে এবং canvas-এ source-pixel box আঁকতে পারেন; page-টি detector proposal দেখায় না। `Save frame` চাপলেই row browser storage-এ থাকে; `Download reviewed JSONL` চাপলে `reviewed_manually: true` ও frame fingerprint-সহ JSONL নামে। কোনো label বা metric স্বয়ংক্রিয়ভাবে তৈরি হয় না, browser data network-এ যায় না।

চালানোর command:

```powershell
.venv\Scripts\python.exe tools\export_review_gallery.py "data\input\videos\vd-1.mp4" --shortlist-csv "reports\vd-1_manual_review_shortlist.csv" --output-dir "data\previews\vd-1_review_gallery"
```

Runtime run-এ ১২/১২ frame 2560×1440-এ export হয়েছে; manifest-এ video SHA-256 `3917b7d5…` ও প্রতি source frame fingerprint আছে। `labels_saved` এখনো false এবং `vd-1_user_verified.jsonl` file নেই। HTML ও JPEG `data/previews/`-এ থাকায় Git-ignored; source utility ও template tracked হবে। UI screenshot/interaction browser bridge দিয়ে verify করা যায়নি, তাই visual/JavaScript interaction QA বাকি।

Windows-এ gallery খোলার জন্য `tools/start_vd1_review_gallery.bat` আছে। Launcher local video ও virtual environment যাচাই করে, dense tracking report থাকলে shortlist না থাকলে সেটি তৈরি করে, তারপর বর্তমান template দিয়ে gallery refresh করে default browser-এ খোলে। Dense tracking report অনুপস্থিত হলে সেটি বানানোর command দেখায়। Launcher চালানো বা browser-এ page খোলা মানে annotation নয়—প্রতিটি frame মানুষ দেখে save না করা পর্যন্ত label file তৈরি হবে না। এই environment থেকে browser window-র দৃশ্যমানতা এখনো যাচাই করা যায়নি।

### M92: review page-এ unsaved edit রক্ষা

Save-এর আগে অন্য frame-এ যাওয়া বা page বন্ধ করলে box/status edit হারানোর ঝুঁকি ছিল। এখন frame navigation-এ discard confirmation আসে, unsaved edit থাকা অবস্থায় JSONL export থামে, progress-এ current edit দেখা যায় এবং page বন্ধের আগেও browser সতর্ক করে। `Save frame`-ই label store করার একমাত্র action; auto-generated label নেই। Browser interaction QA এখনো user-এর local browser-এ করা বাকি।

### M93: browser review-তে optional body-region point label

ভিডিও pose-region evaluator opponent box-এর ভেতরে per-region manual center point চায়। Browser fallback-এ box/status label ছিল, কিন্তু point mode ছিল না। এখন reviewer box নির্বাচন করে ঐচ্ছিকভাবে head, chest, lower-torso, ও legs center mark করতে পারেন; points box-এর ভেতরে না হলে save হবে না এবং JSONL-এ evaluator-এর `opponent_body_regions_xy` schema-তে যাবে। এগুলো approximate visible-image center, game hitbox বা anatomical ground truth নয়। UI interaction ও evaluator integration runtime-এ এখনো যাচাই করা হয়নি।

### M94: one-click gallery refresh

Generated gallery `data/previews/`-এ ignored থাকায় নতুন checkout বা template update-এর পর পুরোনো/অনুপস্থিত HTML খোলার ঝুঁকি ছিল। Windows launcher এখন `.venv` ও `vd-1.mp4` যাচাই করে; shortlist না থাকলে local dense tracking JSON থেকে তৈরি করে; তারপর বর্তমান exporter/template দিয়ে ১২-frame gallery refresh করে এবং browser-এ খোলে। Dense report না থাকলে প্রয়োজনীয় tracking command দেখায়। এটি শুধু local recorded video পড়ে; browser page বা launcher কোনো annotation নিজে তৈরি করে না। Desktop-এ click করে launch path ও visible window এখনো হাতে QA করা হয়নি।

### M95: fresh clone থেকে Git LFS sample video যাচাই

LFS smudge skip করে normal depth-1 checkout করা আলাদা temporary clone-এ `git lfs fetch --include=data/input/videos/vd-1.mp4 origin main` দিয়ে object নামিয়েছি। Clone-এর LFS cache-এ 212,593,238 bytes পাওয়া গেছে, `git lfs fsck --objects` OK দিয়েছে, `git lfs checkout` দিয়ে video materialize হয়েছে, file size `212593238` byte এবং SHA-256 `3917b7d52fe197c69d9305a698a50ff087e9378f01b3ab8c3f26668e5447c930` project source-র সঙ্গে মিলে গেছে। GitHub-hosted sample video fresh clone-এ পাওয়া যায়—এখন তা যাচাই করা হয়েছে।

### M96: ছোট target-এর জন্য browser gallery zoom

Video source resolution আগের মতো পূর্ণ 2560×1440 থাকে; gallery শুরুতে 50% view-তে দেখায়। ছোট দূরের opponent-এ box ও region point ঠিকভাবে বসানোর সুবিধার জন্য 50–200% zoom slider যোগ করেছি। Zoom শুধু screen-এ দেখার আকার বদলায়; canvas source pixels-এই থাকে, তাই JSONL coordinates evaluator-এর frame size-এর সঙ্গে মেলে। Horizontal/vertical overflow হলে gallery-র image pane scroll করা যাবে। Browser visual QA এখনো বাকি।

### M97: review storage-কে নির্বাচিত sample-এর সঙ্গে বাঁধা

একই source video থেকে shortlist বদলালে browser local storage-তে আগের sample-এর frame labels মিশে export হওয়ার ঝুঁকি ছিল। Storage key-তে এখন ordered frame-index তালিকাও থাকে; তাই আলাদা shortlist আলাদা review state পায়। একই video ও একই shortlist-এ launcher template refresh করলেও সংরক্ষিত manual labels browser storage-তে আগের মতোই থাকে। JSONL export-এ gallery sample-এ থাকা saved rows-ই যাবে।

### M98: downloaded annotation validate করে import

Browser থেকে JSONL download করার পর সঠিক annotation filename/path-এ সরাতে ভুল হতে পারে এবং ভুল video-র label evaluator-এ চলে যেতে পারে। `tools/import_review_annotations.py` এখন JSONL-কে canonical `data/annotations/<video>_user_verified.jsonl` path-এ copy করার আগে manual-review flag, non-prefilled source, shortlist-এর exact frame coverage, video SHA, decoded frame SHA, timestamp, dimensions, box bounds ও optional body-point schema/bounds যাচাই করে। Timestamp-কে decoded source timestamp থেকে 1 ms-এর বেশি বিচ্যুত হতে দেওয়া হয় না। `.venv` দিয়ে `--help` চালিয়ে CLI/import load যাচাই হয়েছে; কোনো review file দেওয়া হয়নি, annotation file তৈরি হয়নি, পূর্ণ import flow test করা হয়নি। Existing destination overwrite করা default-এ নিষিদ্ধ; `--replace` স্পষ্টভাবে দিতে হয়। Importer ground truth বানায় না—downloaded human-reviewed file ছাড়া কাজ করবে না।

### M99: gallery launcher-এর manual-open fallback

Windows shell browser-কে foreground-এ না আনলে launcher এখন local `index.html`-এর absolute path সবসময় দেখায়। এতে browser নিজে না উঠলেও user একই offline gallery path হাতে খুলতে পারবেন। Launcher নিজে কোনো label লেখে না; desktop UI visibility এই environment-এ এখনো যাচাই হয়নি।
