# VantaSight: কাজের ধাপ ও বর্তমান অবস্থা

এটি VantaSight তৈরির ধাপে ধাপে কাজের তালিকা। প্রতিটি ধাপ শুরুর আগে কী করতে যাচ্ছি তা জানাব। ধাপ শেষ হলে কী বদলেছে, কেন দরকার, কীভাবে ব্যবহার করতে হবে এবং কোনো সীমাবদ্ধতা আছে কি না—সহজ ভাষায় বুঝিয়ে দেব। কোনো কাজ যাচাই করা না গেলে সেটিও পরিষ্কার করে বলব।

## প্রকল্পের লক্ষ্য

নিজের screenshot বা recorded gameplay video বিশ্লেষণ করে দৃশ্যমান তথ্য নিয়ে গবেষণা করা: character candidate শনাক্ত করা, দৃশ্যমান body-region অনুমান করা, পরপর frame-এ detection track করা, এবং crosshair reference থেকে target region-এর দূরত্ব মাপা। প্রকল্পটি game input পাঠাবে না, aim নড়াবে না, fire করবে না, game memory পড়বে না বা game process-এ প্রবেশ করবে না। প্রথম সংস্করণ শুধু offline footage বিশ্লেষণ করবে।

## এখন পর্যন্ত কী করা হয়েছে

**অবস্থা: M1-এর প্রাথমিক code লেখা হয়েছে; dependency install ও runtime verification বাকি।**

- `src/vantasight/input/image_loader.py` image file পড়ে pixel data, width, height এবং path দেয়। Windows-এ non-English অক্ষর থাকা path-ও পড়তে পারে। Pixel channel OpenCV-এর BGR format-এ থাকে।
- `src/vantasight/input/video_loader.py` video-র width, height, FPS, frame count ও আনুমানিক duration জানায়। এর `frames()` function প্রতিটি decoded frame-এর zero-based index, timestamp ও pixel data দেয়। `with` block শেষ হলে video file release হয়।
- `src/vantasight/cli.py` terminal-এ image বা video path দিলে তার metadata দেখায়।
- `src/vantasight/__main__.py` দিয়ে `python -m vantasight` command চালানো যায়।
- `requirements.txt` এবং `pyproject.toml` OpenCV dependency ও Python package setup নির্ধারণ করে।
- `README.md`-তে প্রাথমিক setup ও ব্যবহারবিধি আছে। `.gitignore` local footage ও তৈরি হওয়া report Git-এ যোগ হওয়া থেকে বাদ রাখে।

এখনো detector, body-region model, tracker, alignment calculation বা dashboard বানানো হয়নি। বর্তমান session-এ pip offline mode-এ আছে এবং virtual environment তৈরি করার সময় temporary file লেখার permission error হয়েছে। তাই OpenCV install হয়নি, code-ও runtime-এ চালিয়ে যাচাই করা যায়নি।

## কাজের roadmap

| ধাপ | কাজ | ধাপ শেষে যা পাওয়া যাবে | অবস্থা |
|---|---|---|---|
| 1. Input foundation | Python package, screenshot/video reader, metadata | Image ও video load করার ভিত্তি | Code লেখা; install ও runtime verification বাকি |
| 2. Frame preview | Video থেকে নির্দিষ্ট frame বের করা, resize ও coordinate mapping, preview দেখা | বাস্তব footage-এর frame দেখে input pipeline যাচাই | শুরু হয়নি |
| 3. প্রাথমিক detection | উপযুক্ত pretrained detector যোগ করা; confidence ও bounding box দেখানো | Model কোন জিনিস শনাক্ত করছে তা পর্যালোচনা | শুরু হয়নি |
| 4. Dataset review | বাস্তবসম্মত clip-এ false positive ও missed detection হাতে পর্যালোচনা | Detector উপযোগী কি না তার প্রাথমিক প্রমাণ | শুরু হয়নি |
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

এরপর screenshot বা recorded video inspect করতে:

```powershell
python -m vantasight "D:\path\to\screenshot.png"
python -m vantasight "D:\path\to\recording.mp4"
```

Python code থেকে video frame পড়ার উদাহরণ:

```python
from vantasight.input import VideoReader

with VideoReader("recording.mp4") as reader:
    print(reader.info)
    for frame in reader.frames():
        print(frame.index, frame.timestamp_seconds, frame.pixels.shape)
        break  # এই উদাহরণে শুধু প্রথম frame পড়া হয়।
```

## আমরা যেভাবে কাজ করব

প্রতিটি ধাপে:

1. কাজের লক্ষ্য এবং কোন file বা অংশে পরিবর্তন হতে পারে তা আগে জানাব।
2. প্রয়োজনীয় code বা documentation update করব এবং dependency বা permission সমস্যা থাকলে জানাব।
3. ধাপ শেষে কী বদলেছে, কীভাবে ব্যবহার করতে হবে, কী যাচাই হয়েছে এবং কী সীমাবদ্ধতা আছে—তা বুঝিয়ে বলব।
4. বাস্তব footage লাগলে তোমার দেওয়া screenshot বা ছোট recorded clip দিয়ে ফল পর্যালোচনা করব।

## পরের কাজের জন্য যা দরকার

M1 runtime-এ যাচাই করতে OpenCV install করতে হবে। M2 ও M3-তে এগোতে `data/input/` folder-এ একটি screenshot এবং ছোট recorded video রাখলে সুবিধা হবে। Footage ছাড়াও code structure এগোনো যাবে, তবে বাস্তব উদাহরণে detection-এর মান যাচাই করা যাবে না।

## ভাষার নিয়ম

এই workflow নথিটি Bangla-তে থাকবে, যাতে ধাপগুলো অনুসরণ করা সহজ হয়। Code, variable, command ও মূল project documentation আপাতত English-এ থাকবে। Project শেষ হলে প্রয়োজন অনুযায়ী পুরো project-কে consistent international English-এ নেওয়া হবে।
