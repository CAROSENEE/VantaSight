# LinkedIn post draft

## VantaSight | Offline FPS Gameplay Analytics

Can a computer-vision model tell what is actually visible in an FPS frame—and how do we know when it gets that wrong?

I’m building **VantaSight**, an offline research prototype for analyzing recorded FPS gameplay. The current work is focused on making the analysis inspectable and repeatable:

- Load recorded video and inspect its dimensions, frame rate, and duration.
- Sample frames at a rate based on the source video’s FPS.
- Run a baseline person detector and short-term tracker on recorded footage.
- Review gameplay screenshots with source-image checks and compare model results with labels.
- Keep evaluation reports and the analysis dashboard explicit about their sampling settings and limitations.

The current evidence is deliberately modest: a two-screenshot presence pilot is useful for exploring the workflow, but it is far too small to establish gameplay accuracy. Video-level labels are still pending, so I’m not claiming enemy-identification accuracy, reliable tracking, or aim-performance results.

Next, I’m working toward a carefully reviewed set of video frames, then I can measure missed detections and false positives—including how target size affects detection—against those labels.

VantaSight is for offline analysis of recorded footage. It does not control the game, move the mouse, fire, or provide live gameplay assistance.

What would you want to see in a transparent evaluation of a computer-vision system for gameplay footage?

#ComputerVision #OpenSource #MachineLearning #GameAnalytics #Python

## Images to attach

1. **Primary image:** a side-by-side pair using `docs/assets/ss-1_candidate_preview.jpg` and `docs/assets/ss-2_candidate_preview.jpg`. Caption it “Visible in-world opponent” / “No visible in-world opponent.” These illustrate the screenshot-presence pilot only; they are not proof of video accuracy.
2. **Optional second image:** a dashboard screenshot showing the VantaSight title and the source video’s sampling metadata. Capture it locally from the dashboard; blur player names and any account identifiers before posting.
3. **Optional third image:** a review-gallery frame with the review controls visible, clearly showing that video labels are awaiting review. Do not present blank/unreviewed frames as completed annotations.

Use the two supplied gameplay screenshots only after checking that visible player names, match details, and account identifiers are acceptable to share. Crop or blur them if needed. Do not upload a video clip unless you intend that full clip to be public.
