# Cross-Modal Alignment and Modality Independence

Nguon dung de thiet ke alignment khi hinh anh va transcript khong nhat thiet
mo ta cung mot noi dung, vi du talking-head documentary.

| Paper | Link | Vai tro trong thiet ke |
|---|---|---|
| Multimodal Dual Attention Memory for Video Story Question Answering | https://openaccess.thecvf.com/content_ECCV_2018/html/Kyungmin_Kim_Multimodal_Dual_Attention_ECCV_2018_paper.html | Xu ly tung modality bang attention/memory rieng truoc khi late fusion; ung ho viec khong tron visual va transcript qua som. |
| MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis | https://arxiv.org/abs/2005.03545 | Tach thong tin modality-specific khoi phan shared; graph nen giu visual-only va transcript-only facts, chi fuse phan co bang chung chung. |
| The Promise of Premise: Harnessing Question Premises in Visual Question Answering | https://aclanthology.org/D17-1097/ | Cho thay model co the tra loi sai khi text premise khong lien quan den image; can relevance/grounding check truoc khi fuse. |
| Transformer-based Cascaded Multimodal Speech Translation | https://aclanthology.org/2019.iwslt-1.6/ | Incongruence analysis cho thay visual context khong phu hop co the lam giam chat luong; them modality khong phai luc nao cung tot. |
| Multimodal Transformer for Unaligned Multimodal Language Sequences | https://aclanthology.org/P19-1656/ | Cross-modal interaction can duoc hoc/xet theo cap va theo thoi gian, thay vi gia dinh hai stream da aligned. |
| MTAG: Modal-Temporal Attention Graph for Unaligned Human Multimodal Language Sequences | https://aclanthology.org/2021.naacl-main.79/ | Bieu dien modality va temporal interactions bang heterogeneous graph, sau do prune cac interaction khong quan trong. |

## Nguyen tac rut ra

1. Luon giu modality-specific facts.
2. Fusion la tuy chon tren tung entity/relation, khong phai bat buoc tren ca
   segment.
3. `no_merge` la ket qua hop le.
4. Edge `modalities=["visual", "transcript"]` chi ton tai khi co bridge evidence
   duoc kiem tra.
5. Talking-head video van giu day du transcript graph ngay ca khi visual va
   transcript semantic correspondence rat thap.

## Hien thuc trong pipeline

Pipeline dung conditional late fusion train-free:

1. MiniCPM tao visual-only facts ma khong thay transcript.
2. spaCy transcript memory cung cap transcript propositions.
3. OpenCLIP text encoder tinh cosine similarity giua hai tap fact.
4. Chi cap mutual-best vuot threshold va ambiguity margin moi duoc dua cho
   MiniCPM nhu merge candidate.
5. Validator va entity registry, khong phai MiniCPM, thuc hien merge va ghi
   global graph.
