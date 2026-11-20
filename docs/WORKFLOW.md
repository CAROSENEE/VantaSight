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

এখনো detector, body-region model, tracker, alignment calculation বা dashboard তৈরি হয়নি।

## কাজের roadmap

| ধাপ | কাজ | ধাপ শেষে যা পাওয়া যাবে | অবস্থা |
|---|---|---|---|
| 1. Input foundation | Python package, screenshot/video reader, metadata | Image ও video load করার ভিত্তি | `vd-1.mp4` দিয়ে runtime-এ যাচাই হয়েছে |
| 2. Frame preview | Video থেকে নির্দিষ্ট frame বের করা ও preview image save করা | বাস্তব footage-এর নির্দিষ্ট frame দেখা | Frame 3540 সফলভাবে বের ও দেখা হয়েছে |
| 3. প্রাথমিক detection | COCO pretrained person-candidate model; confidence ও bounding box দেখানো | Model কোন জিনিস শনাক্ত করছে তা পর্যালোচনা | প্রথম inference ও per-source mask হয়েছে; false positive বাকি |
| 4. Dataset review | বাস্তবসম্মত clip-এ false positive ও missed detection হাতে পর্যালোচনা | Detector উপযোগী কি না তার প্রাথমিক প্রমাণ | দুই screenshot ও একটি video frame review হয়েছে; বিস্তৃত review বাকি |
| 5. Body-region পরীক্ষা | Pose/region model চেষ্টা করা ও অনিশ্চয়তা দেখানো | Confidence-সহ region estimate | শুরু হয়নি |
| 6. Tracking | একাধিক frame-এর detection মিলিয়ে temporary ID দেওয়া | Track কতটা ধারাবাহিক তা দেখা | শুরু হয়নি |
| 7. Alignment metrics | Crosshair reference সেট করা এবং pixel/normalized distance হিসাব | জ্যামিতিক alignment measurement | শুরু হয়নি |
| 8. Export ও dashboard | CSV/JSON এবং আলাদা analysis preview/dashboard | ব্যবহারযোগ্য research prototype | শুরু হয়নি |
| 9. Evaluation ও documentation | Annotated sample-এ metric মাপা, limitation লেখা | পুনরায় যাচাইযোগ্য ফল ও setup guide | শুরু হয়নি |

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

## ছবি ও ভিডিও GitHub-এ যাবে কি?

`data/input/screenshots/` এবং `data/input/videos/`-এর media Git ignore করা আছে। তাই স্বাভাবিক `git add`, commit ও push-এ ছবিগুলো বা ভিডিও GitHub-এ যাবে না। `git add -f` দিয়ে জোর করে যোগ করলে upload হতে পারে—ইচ্ছা করে না করলে এমন করবে না। `data/previews/`-এর তৈরি preview-ও Git ignore করা।

Folder-গুলো খালি থাকলেও Git-এ দেখানোর জন্য ছোট `.gitkeep` placeholder file রাখা আছে। এগুলোতে ছবি বা ভিডিও থাকে না।

## আমরা যেভাবে কাজ করব

1. কাজ শুরুর আগে লক্ষ্য এবং কোন file বা অংশে পরিবর্তন হতে পারে তা জানাব।
2. প্রয়োজনীয় code বা documentation update করব এবং dependency বা permission সমস্যা থাকলে জানাব।
3. ধাপ শেষে কী বদলেছে, কীভাবে ব্যবহার করতে হবে, কী যাচাই হয়েছে এবং কী সীমাবদ্ধতা আছে—তা বুঝিয়ে বলব।
4. বাস্তব footage লাগলে তোমার দেওয়া screenshot বা ছোট recorded clip দিয়ে ফল পর্যালোচনা করব।

## পরের কাজের জন্য যা দরকার

M1, M2 এবং M3-এর প্রথম runtime inference হয়েছে। Per-source exclusion mask যোগ করে HUD/webcam false positive কিছুটা কমেছে; `ss-2`-এর weapon/arm এবং `vd-1`-এর first-person weapon/arm এখনো ভুল candidate হচ্ছে। পরের কাজ হলো এই false positive-গুলোর জন্য নিরাপদ ও পুনর্ব্যবহারযোগ্য handling করা এবং বিদ্যমান `vd-1`-এর একাধিক frame পরীক্ষা করা। এখনই নতুন file দিতে হবে না। পরে detector-কে ভালোভাবে মাপতে হলে enemy আছে ও নেই—দুই ধরনের representative sample-এ হাতে label লাগবে।

## ভাষার নিয়ম

এই workflow নথিটি Bangla-তে থাকবে, যাতে ধাপগুলো অনুসরণ করা সহজ হয়। Code, variable, command ও মূল project documentation আপাতত English-এ থাকবে। Project শেষ হলে প্রয়োজন অনুযায়ী পুরো project-কে consistent international English-এ নেওয়া হবে।
