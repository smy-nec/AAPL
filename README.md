# Action-Agnostic Point-Level Supervision for Temporal Action Detection

Official repository for our paper accepted for presentation in AAAI-25.

> :page_with_curl:***Action-Agnostic Point-Level Supervision for Temporal Action Detection*** \
> Shuhei M Yoshida, Takashi Shibata, Makoto Terao, Takayuki Okatani, Masashi Sugiyama

## Abstract

We propose action-agnostic point-level (AAPL) supervision for temporal action detection
to achieve accurate action instance detection with a lightly annotated dataset.
In the proposed scheme, a small portion of video frames is sampled in an unsupervised manner and presented to human annotators,
who then label the frames with action categories.
Unlike point-level supervision, which requires annotators to search for every action instance in an untrimmed video,
frames to annotate are selected without human intervention in AAPL supervision.
We also propose a detection model and learning method to effectively utilize the AAPL labels.
Extensive experiments on the variety of datasets (THUMOS '14, FineAction, GTEA, BEOID, and ActivityNet 1.3) demonstrate
that the proposed approach is competitive with or outperforms prior methods for video-level and point-level supervision
in terms of the trade-off between the annotation cost and detection performance.
The code and the annotation tool used in this study are included in the supplementary material
and will be made available to the public if our paper is accepted.

## Citation

If you use this code or find it helpful, please cite our paper:

```text
Yoshida, S. M., Shibata, T., Terao, M., Okatani, T., & Sugiyama, M. (2025). Action-Agnostic Point-Level Supervision for Temporal Action Detection. Proceedings of the AAAI Conference on Artificial Intelligence, 39(9), 9571-9579. https://doi.org/10.1609/aaai.v39i9.33037
```
