| Dataset/encoder | k | graph (dev-selected) | multi_hop - RRF (pp) [95% CI] (p) | multi_hop - cross-encoder rerank (pp) [95% CI] (p) | multi_hop - PPR (pp) [95% CI] (p) |
|---|---|---|---|---|---|
| hotpotqa/minilm | 5 | mention | +16.8 [+13.6, +20.2] (p=<0.001) | +11.4 [+8.1, +14.9] (p=<0.001) | +7.5 [+4.9, +10.0] (p=<0.001; PPR graph=mention) |
| hotpotqa/minilm | 10 | mention | +9.0 [+6.0, +12.1] (p=<0.001) | +13.1 [+10.1, +16.2] (p=<0.001) | +0.1 [-1.4, +1.5] (p=1.000; PPR graph=mention) |
| hotpotqa/minilm | 20 | mention | +3.5 [+0.9, +6.0] (p=0.013) | +17.0 [+14.2, +19.7] (p=<0.001) | +0.0 [-1.1, +1.1] (p=1.000; PPR graph=mention) |
| hotpotqa/bge | 5 | mention | +12.2 [+9.4, +15.3] (p=<0.001) | +9.7 [+6.8, +12.8] (p=<0.001) | +1.7 [-0.6, +3.9] (p=0.167; PPR graph=mention) |
| hotpotqa/bge | 10 | mention | +7.7 [+5.3, +10.2] (p=<0.001) | +7.4 [+5.1, +9.8] (p=<0.001) | -0.4 [-1.4, +0.6] (p=0.549; PPR graph=mention) |
| hotpotqa/bge | 20 | mention | +5.6 [+3.8, +7.4] (p=<0.001) | +7.7 [+6.1, +9.5] (p=<0.001) | +0.0 [-0.9, +0.9] (p=1.000; PPR graph=mention) |
| 2wiki/minilm | 5 | mention+entity | +30.6 [+26.9, +34.2] (p=<0.001) | +30.0 [+26.3, +33.6] (p=<0.001) | +12.0 [+9.6, +14.4] (p=<0.001; PPR graph=mention) |
| 2wiki/minilm | 10 | mention+entity | +39.9 [+36.0, +43.6] (p=<0.001) | +43.5 [+40.0, +47.2] (p=<0.001) | +2.3 [+1.3, +3.4] (p=<0.001; PPR graph=mention) |
| 2wiki/minilm | 20 | mention+entity | +40.6 [+36.9, +44.2] (p=<0.001) | +48.3 [+45.2, +51.6] (p=<0.001) | +1.6 [+0.3, +2.9] (p=0.019; PPR graph=mention) |
| 2wiki/bge | 5 | mention | +39.1 [+35.7, +42.2] (p=<0.001) | +38.4 [+35.1, +41.7] (p=<0.001) | +18.9 [+15.9, +21.9] (p=<0.001; PPR graph=mention) |
| 2wiki/bge | 10 | mention | +49.0 [+45.7, +52.1] (p=<0.001) | +48.6 [+45.4, +51.8] (p=<0.001) | +1.8 [+0.7, +2.9] (p=0.002; PPR graph=mention) |
| 2wiki/bge | 20 | mention | +47.0 [+43.8, +50.1] (p=<0.001) | +46.3 [+43.3, +49.5] (p=<0.001) | +0.9 [+0.4, +1.5] (p=0.004; PPR graph=mention) |
| musique/minilm | 5 | mention | +5.8 [+3.7, +8.0] (p=<0.001) | +0.6 [-1.6, +2.9] (p=0.672) | +0.1 [-1.6, +1.7] (p=1.000; PPR graph=mention) |
| musique/minilm | 10 | mention | +7.0 [+4.4, +9.8] (p=<0.001) | +3.7 [+1.2, +6.4] (p=0.005) | -1.5 [-3.3, +0.2] (p=0.120; PPR graph=mention) |
| musique/minilm | 20 | mention | +5.8 [+2.6, +9.0] (p=<0.001) | +8.5 [+5.6, +11.5] (p=<0.001) | -0.4 [-1.9, +1.1] (p=0.679; PPR graph=mention) |
| musique/bge | 5 | mention | +6.7 [+4.5, +9.2] (p=<0.001) | +2.7 [+0.2, +5.5] (p=0.046) | +1.4 [-0.1, +2.9] (p=0.091; PPR graph=mention) |
| musique/bge | 10 | mention | +6.3 [+3.5, +9.2] (p=<0.001) | +5.0 [+2.4, +7.7] (p=<0.001) | +0.5 [-1.0, +2.0] (p=0.624; PPR graph=mention) |
| musique/bge | 20 | mention | +5.8 [+2.7, +8.9] (p=<0.001) | +7.1 [+4.2, +9.7] (p=<0.001) | -0.5 [-2.0, +1.0] (p=0.611; PPR graph=mention) |