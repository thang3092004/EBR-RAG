# Entity-Anchored TM Graph: Paper Index

Day la danh sach nguon ban dau cho huong cai tien:

> Train-free entity anchoring before long-video graph extraction.

Muc tieu khong phai claim minh phat minh detector/tracker moi, ma claim minh
de xuat cach dung cac observation tu video/audio de gan global entity IDs, sau
do dua entity IDs vao TM Graph de giam ambiguity nhu `he/she/it`, duplicate
nodes, va sai coreference.

## Baseline video RAG

| Paper | Link | Vai tro trong do an |
|---|---|---|
| VideoRAG: Retrieval-Augmented Generation over Video Corpus | https://arxiv.org/abs/2502.01549 | Baseline chinh. Do an cai tien pipeline graph/video QA cua bai nay. |

## Long-video identity, character grounding, and person clustering

| Paper | Link | Vai tro trong do an |
|---|---|---|
| Face, Body, Voice: Video Person-Clustering with Multiple Modalities | https://arxiv.org/abs/2105.09939 | Blueprint rat gan voi bai toan: gom person identity trong video bang face, body, voice, va constraints. |
| Character Grounding and Re-Identification in Story of Videos and Text Descriptions | https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500528.pdf | Lien ket nhan vat trong video voi text description; huu ich de bao ve y tuong entity ID + text mention grounding. |
| MovieNet: A Holistic Dataset for Movie Understanding | https://arxiv.org/abs/2007.10937 | Schema/dataset movie-scale gom shot, scene, character, subtitle, description. Huu ich cho long-video structure. |
| DramaQA: Character-centered Video Story Understanding with Hierarchical QA | https://arxiv.org/abs/2005.03356 | Chung minh QA video dai can character-centered annotations, bbox, behavior, emotion, va script coreference. |
| TVQA+: Spatio-Temporal Grounding for Video Question Answering | https://arxiv.org/abs/1904.11574 | Lien he QA voi spatio-temporal grounding; huu ich cho evaluation va motivation. |

## Multiple object tracking and local tracklets

| Paper | Link | Vai tro trong do an |
|---|---|---|
| ByteTrack: Multi-Object Tracking by Associating Every Detection Box | https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf | MOT baseline thuc dung de tao local tracklets tu detections. |
| BoT-SORT: Robust Associations Multi-Pedestrian Tracking | https://arxiv.org/abs/2206.14651 | Tracker manh cho person; co camera motion compensation va ReID. |
| OC-SORT: Observation-Centric SORT | https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html | Tracker don gian/manh khi occlusion va noisy detections; dung lam baseline tracking khac. |

## Generic object / animal tracking and video object memory

| Paper | Link | Vai tro trong do an |
|---|---|---|
| XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model | https://arxiv.org/abs/2207.07115 | Rat phu hop ve y tuong memory dai han cho object segmentation ma khong phinh memory qua muc. |
| SAM 2: Segment Anything in Images and Videos | https://arxiv.org/abs/2408.00714 | Nen tang segmentation/tracking cho object bat ky trong video. |
| DEVA: Tracking Anything with Decoupled Video Segmentation | https://arxiv.org/abs/2309.03903 | Tracking/segmentation object tong quat; huu ich cho entity types ngoai person. |
| LVOS: A Benchmark for Long-term Video Object Segmentation | https://arxiv.org/abs/2211.10181 | Motivation cho kho khan cua object tracking trong video dai. |
| LVOS v2: Long-term Video Object Segmentation Benchmark | https://arxiv.org/abs/2404.19326 | Ban cap nhat cua benchmark long-term VOS. |

## Audio, diarization, and active speaker

| Paper | Link | Vai tro trong do an |
|---|---|---|
| AVA-ActiveSpeaker: An Audio-Visual Dataset for Active Speaker Detection | https://arxiv.org/abs/1901.01342 | Nen tang cho viec noi `SPEAKER_ID` voi `PERSON_ID` khi nguoi trong khung hinh dang noi. |
| AVA-AVD: Audio-Visual Diarization in the Wild | https://arxiv.org/abs/2111.14448 | Huu ich de xu ly "who spoke when", ke ca off-screen speaker. |
| Audiovisual Speaker Diarization of TV Series | https://arxiv.org/abs/1812.07205 | Cach ket hop audio diarization va visual diarization bang matching toi uu. |
| Active Speaker Faces for Diarization in TV Shows | https://arxiv.org/abs/2203.15961 | Lien ket face tracks voi speaker diarization trong TV/video dai. |

## Video scene graph and visual grounding

| Paper | Link | Vai tro trong do an |
|---|---|---|
| Panoptic Video Scene Graph Generation | https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Panoptic_Video_Scene_Graph_Generation_CVPR_2023_paper.html | Lien he entity/object tracking voi scene graph generation. |
| Motion-aware Temporal Panoptic Scene Graph | https://ojs.aaai.org/index.php/AAAI/article/view/32665 | Scene graph co nhan thuc chuyen dong; dung lam related work, khong nham voi temporal edges cua code hien tai. |
| Comprehensive Visual Grounding for Video Description | https://ojs.aaai.org/index.php/AAAI/article/view/28032 | Lien he video description voi visual grounding; huu ich cho ID-aware captioning. |

## Ghi chu citation

Khi dua vao quyen do an, nen chia citation theo vai tro:

1. Baseline long-video RAG: `VideoRAG`.
2. Van de entity ambiguity/coreference: character grounding, DramaQA, TVQA+.
3. Giai phap observation-level: MOT, VOS, audio diarization.
4. Dong gop cua minh: train-free constrained entity linking + entity-anchored
   temporary memory graph.
