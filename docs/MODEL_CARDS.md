# Baseline detector model card

## TorchVision Faster R-CNN MobileNetV3-Large FPN

- **Purpose:** Baseline person-candidate detection on offline screenshots and selected video frames.
- **Weights:** `FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`, downloaded from the official [TorchVision checkpoint URL](https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth) when first used and cached under the ignored `models/cache/` folder.
- **Training label:** COCO `person`. This model does not know Valorant teams and does not identify enemies. Detection confidence is not hit probability.
- **Reported benchmark:** TorchVision reports 32.8 box mAP on COCO val2017 for these weights. This is not a VantaSight or Valorant result.
- **Model size:** TorchVision lists 19,386,354 parameters and a 74.2 MB checkpoint.
- **Code license:** TorchVision is BSD-3-Clause. The repository keeps its GPL-3.0 license.
- **Weight/data terms:** TorchVision cautions that pretrained model weights may have additional terms derived from their training dataset. These weights are COCO-pretrained. COCO's official terms say its annotations are CC BY 4.0 and that image use must follow the original Flickr terms. Review the upstream model/dataset terms before redistributing the checkpoint; VantaSight will not commit model weights.
- **Known limitations:** COCO is made of ordinary photographs, not Valorant renderings. The model may miss stylized, distant, occluded, or effect-obscured characters and may detect replay silhouettes or people in stream overlays as person candidates. These outputs require human review.
- **Resolution:** Input frames remain at their original resolution and output boxes are mapped to that coordinate space. The model performs its own configurable resize internally; later evaluation must include small targets and report misses.

Sources:

- [TorchVision model documentation](https://docs.pytorch.org/vision/stable/models/generated/fasterrcnn_mobilenet_v3_large_fpn.html)
- [TorchVision source license](https://github.com/pytorch/vision/blob/main/LICENSE)
- [TorchVision pretrained-weight terms notice](https://docs.pytorch.org/vision/stable/models.html)
- [COCO official terms of use](https://cocodataset.org/dataset/termsofuse.htm)
- [Ultralytics license terms (considered and not selected)](https://www.ultralytics.com/license)

## Experimental body-region estimate: TorchVision Keypoint R-CNN

- **Purpose:** Offline feasibility experiment for visible human keypoints and coarse head/chest/hip/leg reference points.
- **Weights:** `KeypointRCNN_ResNet50_FPN_Weights.COCO_V1`, downloaded from the [official TorchVision checkpoint URL](https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth) when first used and cached under ignored `models/cache/`.
- **Training labels:** COCO person boxes and 17 human keypoints. The model does not know Valorant teams or enemy identity; derived body-region centers are geometric summaries of visible keypoints, not anatomical classification.
- **Reported benchmark:** TorchVision lists 65.0 keypoint mAP on COCO val2017. This is not a VantaSight or game-footage result.
- **Model size:** 59,137,258 parameters, 137.42 GFLOPS, and a 226.1 MB checkpoint according to TorchVision.
- **Code license and weight caveat:** TorchVision is BSD-3-Clause. COCO-derived weights may carry additional dataset terms; see the COCO note above before redistributing the checkpoint. VantaSight does not commit weights.
- **Known limitations:** COCO pose examples are ordinary photographs. Small or stylized game characters, animation, occlusion, and effects may produce missing or incorrect keypoints. Low-confidence and absent points must remain uncertain; never fill them with guessed coordinates.

Sources:

- [TorchVision Keypoint R-CNN documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html)
- [TorchVision model source license](https://github.com/pytorch/vision/blob/main/LICENSE)
