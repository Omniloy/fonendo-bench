# fonendo-bench leaderboard

Generated 2026-09-15 by `fonendo report` (fonendo 0.1.0.dev0, normalizer `es1+lc1`). Every system except the results-only rows was evaluated in its **default configuration**: no custom vocabulary, keyterms or context prompt; Spanish selected where the system allows it; instruction-following models get only the fixed transcription instruction they need. That covers the systems run with this package and the open-weights rows without a runner (`--model` –), which Omniloy ran outside the package on the same audio, with the same scoring and the settings listed under *Systems*. Results-only rows (¹) were evaluated by their owner under the conditions given in their note.

Values are percentages (insertions: per 1,000 reference words), with the 95% bootstrap interval in small type (2,000 resamples, seed 0; clinical: whole sentences resampled, real speech: clips resampled). Lower is better except for term recall. Results-only rows are pinned to the top of each table; the other rows are sorted by WER (real speech: by mean WER). Neighbouring rows whose intervals overlap may not differ, so use `fonendo compare` for a paired test before calling one system better than another.

## Clinical dictation (`clinical_test`, 300 clips)

| System | Type | WER | Term recall | B-WER (term words) | U-WER (other words) | Insertions / 1k | Degenerate |
|---|---|--:|--:|--:|--:|--:|--:|
| **OmniScribe 2 (Omniloy, self-hosted, not publicly available)** ¹ | results only | 8.5 <sub>6.9–10.3</sub> | 85.7 <sub>81.9–89.4</sub> | 8.2 <sub>5.6–11.0</sub> | 8.6 <sub>7.0–10.3</sub> | 10.3 <sub>7.3–13.7</sub> | 0.0 |
| Soniox stt-rt-v5 | commercial API | 7.2 <sub>6.0–8.5</sub> | 70.8 <sub>65.5–76.2</sub> | 17.0 <sub>13.3–21.0</sub> | 5.6 <sub>4.5–6.8</sub> | 9.6 <sub>6.7–12.5</sub> | 0.0 |
| Voxtral Small 24B (FP8 weights) | open weights | 8.5 <sub>6.7–10.3</sub> | 70.3 <sub>64.6–75.7</sub> | 17.0 <sub>13.2–21.1</sub> | 7.1 <sub>5.5–8.8</sub> | 14.1 <sub>9.8–19.2</sub> | 0.0 |
| Cohere Transcribe | open weights | 9.3 <sub>7.8–10.9</sub> | 60.4 <sub>54.6–66.4</sub> | 21.5 <sub>17.6–25.7</sub> | 7.4 <sub>6.0–8.9</sub> | 12.0 <sub>8.9–15.4</sub> | 0.0 |
| Whisper large-v3-turbo | open weights | 10.3 <sub>8.4–12.3</sub> | 63.8 <sub>58.1–69.4</sub> | 21.5 <sub>17.5–25.9</sub> | 8.5 <sub>6.7–10.4</sub> | 15.4 <sub>10.5–21.6</sub> | 0.0 |
| Whisper large-v3 | open weights | 10.6 <sub>8.6–12.7</sub> | 64.1 <sub>58.1–69.9</sub> | 21.2 <sub>17.0–25.7</sub> | 8.9 <sub>7.0–11.0</sub> | 16.3 <sub>11.3–22.1</sub> | 0.3 |
| Deepgram Nova-3 (es) | commercial API | 11.5 <sub>9.5–13.6</sub> | 62.8 <sub>56.7–68.7</sub> | 22.7 <sub>18.4–27.4</sub> | 9.7 <sub>7.9–11.7</sub> | 14.3 <sub>10.7–18.0</sub> | 0.3 |
| Canary-1B-v2 | open weights | 11.8 <sub>9.8–13.9</sub> | 56.0 <sub>49.7–62.0</sub> | 25.5 <sub>21.1–30.3</sub> | 9.6 <sub>7.7–11.7</sub> | 18.2 <sub>12.7–25.2</sub> | 0.0 |
| Gemma 4 E4B | open weights | 11.9 <sub>9.8–14.0</sub> | 63.3 <sub>57.4–69.0</sub> | 24.1 <sub>19.6–29.1</sub> | 10.0 <sub>8.1–11.9</sub> | 12.7 <sub>9.3–16.7</sub> | 0.0 |
| Parakeet-TDT v3 | open weights | 12.3 <sub>10.1–14.6</sub> | 58.6 <sub>52.6–64.4</sub> | 25.4 <sub>20.9–30.3</sub> | 10.2 <sub>8.1–12.5</sub> | 12.7 <sub>9.7–16.0</sub> | 1.0 |
| Voxtral Mini 4B Realtime | open weights | 13.9 <sub>11.2–16.6</sub> | 59.9 <sub>54.1–65.7</sub> | 25.7 <sub>21.0–30.7</sub> | 12.0 <sub>9.4–14.6</sub> | 13.8 <sub>10.2–17.8</sub> | 2.3 |
| Hojo-ASR-Multi-V1 | open weights | 14.0 <sub>11.8–16.3</sub> | 56.5 <sub>50.7–62.4</sub> | 25.5 <sub>21.2–30.4</sub> | 12.1 <sub>10.1–14.3</sub> | 22.1 <sub>16.4–29.2</sub> | 0.0 |
| Whisper large-v3 LoS | open weights | 15.0 <sub>12.9–17.4</sub> | 52.1 <sub>45.7–57.9</sub> | 29.0 <sub>24.5–34.1</sub> | 12.8 <sub>10.8–14.9</sub> | 23.0 <sub>18.3–27.8</sub> | 0.0 |
| omniASR-LLM-7B | open weights | 15.2 <sub>12.8–18.2</sub> | 49.7 <sub>44.3–55.3</sub> | 31.3 <sub>27.0–36.0</sub> | 12.7 <sub>10.2–15.6</sub> | 29.4 <sub>19.1–44.9</sub> | 0.7 |
| Granite Speech 4.1 2B NAR | open weights | 15.4 <sub>13.3–17.4</sub> | 43.8 <sub>37.5–50.0</sub> | 34.2 <sub>29.5–39.3</sub> | 12.4 <sub>10.4–14.4</sub> | 10.3 <sub>7.8–12.8</sub> | 0.0 |
| MOSS-Transcribe-Diarize | open weights | 15.7 <sub>13.3–18.1</sub> | 48.7 <sub>42.3–54.5</sub> | 30.8 <sub>26.0–36.0</sub> | 13.2 <sub>11.0–15.4</sub> | 23.3 <sub>18.0–29.1</sub> | 0.7 |
| Nemotron 3.5 ASR Streaming | open weights | 17.0 <sub>14.8–19.2</sub> | 44.0 <sub>38.0–49.6</sub> | 35.8 <sub>31.3–41.1</sub> | 14.0 <sub>11.8–16.1</sub> | 19.1 <sub>14.9–23.7</sub> | 1.3 |
| Deepgram Flux Multilingual | commercial API | 17.7 <sub>15.0–20.4</sub> | 49.7 <sub>43.5–55.8</sub> | 32.7 <sub>27.8–37.7</sub> | 15.3 <sub>12.7–18.0</sub> | 18.7 <sub>14.3–23.9</sub> | 3.0 |
| Granite Speech 4.1 2B | open weights | 19.2 <sub>16.5–22.0</sub> | 46.9 <sub>41.1–52.5</sub> | 32.1 <sub>27.4–37.1</sub> | 17.1 <sub>14.4–20.1</sub> | 22.1 <sub>17.1–27.6</sub> | 1.3 |
| Whisper large-v3 clinical-assistance | open weights | 20.2 <sub>14.3–30.1</sub> | 55.5 <sub>49.7–61.1</sub> | 28.2 <sub>23.5–33.2</sub> | 18.9 <sub>12.4–30.0</sub> | 61.9 <sub>19.1–142.6</sub> | 0.3 |
| Canary-1B-flash | open weights | 20.5 <sub>17.6–23.6</sub> | 47.1 <sub>41.3–53.1</sub> | 33.1 <sub>28.6–38.3</sub> | 18.5 <sub>15.6–21.6</sub> | 27.8 <sub>22.0–33.6</sub> | 6.3 |
| Phi-4-multimodal-instruct | open weights | 24.0 <sub>10.4–49.8</sub> | 59.1 <sub>52.7–65.2</sub> | 24.8 <sub>20.2–29.6</sub> | 23.9 <sub>8.5–53.6</sub> | 133.5 <sub>12.9–381.4</sub> | 0.3 |
| Granite Speech 4.1 2B Plus | open weights | 26.3 <sub>21.1–34.2</sub> | 38.8 <sub>32.7–44.8</sub> | 40.1 <sub>34.8–45.4</sub> | 24.1 <sub>18.4–33.0</sub> | 59.6 <sub>27.3–122.0</sub> | 1.0 |
| Parakeet-RNNT 1.1B es (projecte-aina) | open weights | 27.8 <sub>24.0–31.6</sub> | 40.6 <sub>35.0–46.2</sub> | 40.2 <sub>35.1–45.6</sub> | 25.8 <sub>21.9–29.7</sub> | 11.7 <sub>8.4–15.1</sub> | 13.0 |
| Voxtral Mini 3B | open weights | 29.0 <sub>9.7–67.8</sub> | 57.8 <sub>51.5–63.8</sub> | 24.1 <sub>19.8–28.9</sub> | 29.7 <sub>7.7–74.4</sub> | 194.4 <sub>13.7–571.1</sub> | 0.3 |
| VibeVoice-ASR | open weights | 67.7 <sub>21.9–130.7</sub> | 50.0 <sub>43.5–55.9</sub> | 44.7 <sub>28.1–73.4</sub> | 71.3 <sub>19.0–144.8</sub> | 504.2 <sub>57.0–1130.7</sub> | 1.7 |

### Clean and degraded audio

Clean: 147 clips. Degraded: 153 clips with added noise (20 to 5 dB SNR) and, on many clips, a phone or low-bitrate codec, room reverberation or a speed or pitch change. The two groups are mostly different sentences.

| System | WER clean | WER degraded | Term recall clean | Term recall degraded |
|---|--:|--:|--:|--:|
| **OmniScribe 2 (Omniloy, self-hosted, not publicly available)** ¹ | 3.7 <sub>2.8–4.6</sub> | 13.3 <sub>10.5–16.4</sub> | 92.7 <sub>88.9–96.0</sub> | 78.6 <sub>72.7–84.4</sub> |
| Soniox stt-rt-v5 | 4.8 <sub>3.7–6.0</sub> | 9.5 <sub>7.5–11.6</sub> | 76.6 <sub>69.7–83.4</sub> | 65.1 <sub>57.8–72.1</sub> |
| Voxtral Small 24B (FP8 weights) | 4.0 <sub>3.1–5.0</sub> | 12.8 <sub>9.7–16.1</sub> | 79.2 <sub>72.0–85.5</sub> | 61.5 <sub>53.9–68.6</sub> |
| Cohere Transcribe | 5.2 <sub>4.1–6.3</sub> | 13.4 <sub>10.8–16.2</sub> | 71.4 <sub>64.0–78.2</sub> | 49.5 <sub>41.3–57.3</sub> |
| Whisper large-v3-turbo | 4.8 <sub>3.8–5.8</sub> | 15.6 <sub>12.4–19.0</sub> | 77.6 <sub>70.8–84.0</sub> | 50.0 <sub>42.3–57.5</sub> |
| Whisper large-v3 | 5.2 <sub>4.2–6.3</sub> | 15.9 <sub>12.4–19.6</sub> | 74.0 <sub>66.7–80.3</sub> | 54.2 <sub>46.0–61.9</sub> |
| Deepgram Nova-3 (es) | 5.6 <sub>4.4–6.8</sub> | 17.2 <sub>13.9–20.9</sub> | 76.0 <sub>69.0–82.9</sub> | 49.5 <sub>41.4–57.8</sub> |
| Canary-1B-v2 | 5.5 <sub>4.3–6.7</sub> | 18.0 <sub>14.6–21.8</sub> | 66.7 <sub>58.4–74.3</sub> | 45.3 <sub>37.4–53.4</sub> |
| Gemma 4 E4B | 5.3 <sub>4.2–6.4</sub> | 18.4 <sub>14.8–22.4</sub> | 72.9 <sub>65.2–80.0</sub> | 53.6 <sub>45.8–61.9</sub> |
| Parakeet-TDT v3 | 6.2 <sub>4.6–8.2</sub> | 18.4 <sub>14.6–22.2</sub> | 69.8 <sub>61.5–77.6</sub> | 47.4 <sub>40.0–55.0</sub> |
| Voxtral Mini 4B Realtime | 5.8 <sub>4.2–7.6</sub> | 21.8 <sub>17.3–26.6</sub> | 72.4 <sub>64.5–79.5</sub> | 47.4 <sub>39.6–55.2</sub> |
| Hojo-ASR-Multi-V1 | 7.8 <sub>6.5–9.1</sub> | 20.1 <sub>16.1–24.4</sub> | 63.0 <sub>55.1–71.0</sub> | 50.0 <sub>41.5–58.6</sub> |
| Whisper large-v3 LoS | 8.0 <sub>6.4–9.5</sub> | 21.9 <sub>18.4–25.7</sub> | 64.1 <sub>56.0–72.3</sub> | 40.1 <sub>32.4–47.4</sub> |
| omniASR-LLM-7B | 7.8 <sub>6.4–9.2</sub> | 22.5 <sub>18.2–27.9</sub> | 58.3 <sub>51.1–66.3</sub> | 41.1 <sub>33.3–48.1</sub> |
| Granite Speech 4.1 2B NAR | 8.8 <sub>7.5–10.1</sub> | 21.9 <sub>18.4–25.6</sub> | 53.1 <sub>44.9–61.5</sub> | 34.4 <sub>26.7–41.8</sub> |
| MOSS-Transcribe-Diarize | 9.0 <sub>7.1–11.2</sub> | 22.2 <sub>18.4–26.1</sub> | 60.9 <sub>53.0–68.8</sub> | 36.5 <sub>28.9–44.4</sub> |
| Nemotron 3.5 ASR Streaming | 9.8 <sub>8.3–11.2</sub> | 24.0 <sub>20.3–27.9</sub> | 55.7 <sub>47.6–63.5</sub> | 32.3 <sub>24.6–40.0</sub> |
| Deepgram Flux Multilingual | 7.3 <sub>6.0–8.5</sub> | 27.9 <sub>23.3–33.0</sub> | 64.1 <sub>56.6–71.5</sub> | 35.4 <sub>27.9–42.8</sub> |
| Granite Speech 4.1 2B | 10.0 <sub>8.5–11.5</sub> | 28.2 <sub>23.4–33.1</sub> | 55.7 <sub>47.8–63.7</sub> | 38.0 <sub>30.3–45.5</sub> |
| Whisper large-v3 clinical-assistance | 16.7 <sub>7.1–34.7</sub> | 23.6 <sub>19.4–28.0</sub> | 66.7 <sub>59.5–73.9</sub> | 44.3 <sub>36.1–52.4</sub> |
| Canary-1B-flash | 9.8 <sub>8.3–11.4</sub> | 31.0 <sub>26.0–36.5</sub> | 59.9 <sub>52.1–67.5</sub> | 34.4 <sub>27.1–41.8</sub> |
| Phi-4-multimodal-instruct | 6.7 <sub>5.5–7.8</sub> | 41.0 <sub>14.3–92.5</sub> | 66.1 <sub>58.8–73.4</sub> | 52.1 <sub>43.9–60.2</sub> |
| Granite Speech 4.1 2B Plus | 12.7 <sub>11.1–14.3</sub> | 39.5 <sub>30.3–54.1</sub> | 50.0 <sub>41.4–58.7</sub> | 27.6 <sub>20.0–34.8</sub> |
| Parakeet-RNNT 1.1B es (projecte-aina) | 11.3 <sub>9.9–12.8</sub> | 43.9 <sub>37.6–50.1</sub> | 53.6 <sub>45.9–61.5</sub> | 27.6 <sub>20.9–34.7</sub> |
| Voxtral Mini 3B | 6.6 <sub>5.3–7.9</sub> | 50.8 <sub>13.0–126.4</sub> | 65.1 <sub>57.1–73.2</sub> | 50.5 <sub>42.3–58.4</sub> |
| VibeVoice-ASR | 7.2 <sub>5.9–8.6</sub> | 126.7 <sub>36.5–237.4</sub> | 66.7 <sub>59.3–73.5</sub> | 33.3 <sub>26.0–41.3</sub> |

## Real Spanish speech (word error rate)

Human speakers: read Wikipedia sentences (FLEURS), European Parliament speeches (VoxPopuli) and health-related broadcast media (MediaSpeech). The mean is the unweighted average of the three WERs.

| System | Type | Mean WER | FLEURS (300) | VoxPopuli (200) | MediaSpeech (200) |
|---|---|--:|--:|--:|--:|
| **OmniScribe 2 (Omniloy, self-hosted, not publicly available)** ¹ | results only | 6.6 | 2.9 | 8.3 | 8.4 |
| omniASR-LLM-7B | open weights | 6.5 <sub>5.8–7.3</sub> | 3.4 <sub>2.8–4.1</sub> | 7.7 <sub>6.0–9.7</sub> | 8.5 <sub>7.6–9.4</sub> |
| Voxtral Small 24B (FP8 weights) | open weights | 6.6 <sub>6.0–7.2</sub> | 2.1 <sub>1.7–2.5</sub> | 7.6 <sub>6.1–9.2</sub> | 10.2 <sub>9.3–11.1</sub> |
| Cohere Transcribe | open weights | 6.8 <sub>6.2–7.4</sub> | 3.3 <sub>2.8–3.8</sub> | 6.6 <sub>5.3–8.0</sub> | 10.5 <sub>9.6–11.5</sub> |
| Voxtral Mini 3B | open weights | 7.0 <sub>6.5–7.7</sub> | 3.0 <sub>2.5–3.5</sub> | 8.0 <sub>6.6–9.7</sub> | 10.1 <sub>9.2–11.1</sub> |
| Canary-1B-v2 | open weights | 7.1 <sub>6.5–7.7</sub> | 3.1 <sub>2.6–3.7</sub> | 6.6 <sub>5.3–7.9</sub> | 11.6 <sub>10.6–12.8</sub> |
| Hojo-ASR-Multi-V1 | open weights | 7.3 <sub>6.7–8.0</sub> | 3.0 <sub>2.5–3.5</sub> | 9.0 <sub>7.4–10.7</sub> | 9.9 <sub>9.0–10.9</sub> |
| Soniox stt-rt-v5 | commercial API | 7.4 <sub>6.7–8.2</sub> | 3.3 <sub>2.7–3.8</sub> | 9.1 <sub>7.5–11.1</sub> | 9.8 <sub>8.8–10.7</sub> |
| Granite Speech 4.1 2B NAR | open weights | 7.6 <sub>7.0–8.2</sub> | 3.9 <sub>3.3–4.5</sub> | 8.2 <sub>6.7–9.8</sub> | 10.7 <sub>9.7–11.6</sub> |
| Gemma 4 E4B | open weights | 7.7 <sub>7.1–8.5</sub> | 3.2 <sub>2.7–3.7</sub> | 10.0 <sub>8.2–12.0</sub> | 10.1 <sub>9.2–11.0</sub> |
| Nemotron 3.5 ASR Streaming | open weights | 7.9 <sub>7.2–8.7</sub> | 4.1 <sub>3.5–4.7</sub> | 9.7 <sub>7.8–11.9</sub> | 9.9 <sub>9.0–10.9</sub> |
| Deepgram Flux Multilingual | commercial API | 8.0 <sub>7.3–8.7</sub> | 4.5 <sub>3.7–5.2</sub> | 10.6 <sub>8.9–12.5</sub> | 8.9 <sub>8.0–9.9</sub> |
| Deepgram Nova-3 (es) | commercial API | 8.0 <sub>7.3–8.7</sub> | 5.2 <sub>4.5–5.9</sub> | 10.0 <sub>8.2–11.8</sub> | 8.8 <sub>8.0–9.7</sub> |
| Voxtral Mini 4B Realtime | open weights | 8.1 <sub>7.1–9.2</sub> | 3.1 <sub>2.5–3.7</sub> | 10.7 <sub>8.2–13.6</sub> | 10.5 <sub>9.2–12.1</sub> |
| Whisper large-v3-turbo | open weights | 9.4 <sub>8.1–10.9</sub> | 3.1 <sub>2.5–3.6</sub> | 13.5 <sub>10.0–17.5</sub> | 11.6 <sub>10.1–13.5</sub> |
| Phi-4-multimodal-instruct | open weights | 10.0 <sub>7.0–15.2</sub> | 3.1 <sub>2.6–3.7</sub> | 7.9 <sub>6.4–9.6</sub> | 19.0 <sub>10.4–34.3</sub> |
| Whisper large-v3 | open weights | 10.3 <sub>8.1–13.3</sub> | 2.9 <sub>2.4–3.4</sub> | 10.8 <sub>8.0–14.2</sub> | 17.3 <sub>11.9–26.3</sub> |
| Whisper large-v3 clinical-assistance | open weights | 10.6 <sub>8.4–14.0</sub> | 3.6 <sub>3.0–4.2</sub> | 14.1 <sub>8.2–24.3</sub> | 14.0 <sub>12.0–16.4</sub> |
| Parakeet-TDT v3 | open weights | 10.8 <sub>9.6–12.2</sub> | 3.3 <sub>2.8–3.9</sub> | 7.8 <sub>6.2–9.7</sub> | 21.4 <sub>18.0–25.0</sub> |
| Granite Speech 4.1 2B | open weights | 11.5 <sub>10.2–12.9</sub> | 4.3 <sub>3.7–4.9</sub> | 8.8 <sub>7.1–10.8</sub> | 21.4 <sub>18.3–24.7</sub> |
| MOSS-Transcribe-Diarize | open weights | 12.0 <sub>10.5–13.7</sub> | 3.5 <sub>3.0–4.1</sub> | 21.6 <sub>17.5–26.5</sub> | 10.9 <sub>9.4–12.8</sub> |
| VibeVoice-ASR | open weights | 12.6 <sub>7.7–21.9</sub> | 3.4 <sub>2.7–4.1</sub> | 10.3 <sub>8.5–12.4</sub> | 24.1 <sub>9.9–49.5</sub> |
| Whisper large-v3 LoS | open weights | 12.8 <sub>10.6–15.6</sub> | 4.5 <sub>3.3–6.3</sub> | 8.7 <sub>7.1–10.5</sub> | 25.2 <sub>18.5–33.2</sub> |
| Granite Speech 4.1 2B Plus | open weights | 14.6 <sub>9.4–24.6</sub> | 5.1 <sub>4.4–6.0</sub> | 9.3 <sub>7.7–11.0</sub> | 29.3 <sub>14.2–58.4</sub> |
| Canary-1B-flash | open weights | 17.1 <sub>15.1–19.3</sub> | 4.4 <sub>3.5–5.3</sub> | 8.0 <sub>6.4–9.9</sub> | 38.9 <sub>33.2–44.6</sub> |
| Parakeet-RNNT 1.1B es (projecte-aina) | open weights | 17.2 <sub>15.8–18.8</sub> | 6.1 <sub>5.2–6.9</sub> | 16.3 <sub>13.6–19.5</sub> | 29.3 <sub>26.1–32.7</sub> |

## Systems

License of each model, the `fonendo run --model` name of its runner and the settings of the published run.

| System | Type | License | `--model` | Settings |
|---|---|---|---|---|
| **OmniScribe 2 (Omniloy, self-hosted, not publicly available)** ¹ | results only | proprietary | – | Omniloy's self-hosted clinical transcription system |
| Soniox stt-rt-v5 | commercial API | commercial API | `soniox_stt_rt_v5` | real-time WebSocket API (EU), language hint es (strict), audio at 1x |
| Voxtral Small 24B (FP8 weights) | open weights | Apache-2.0 | `voxtral_small_24b` (experimental) | vLLM, FP8 weight-only quantization, transcription mode with language es |
| Cohere Transcribe | open weights | Apache-2.0 (gated) | `cohere_transcribe` | bf16, greedy, language es |
| Whisper large-v3-turbo | open weights | MIT | `whisper_large_v3_turbo` | fp16, greedy, temperature 0 without fallback, language es |
| Whisper large-v3 | open weights | Apache-2.0 | `whisper_large_v3` | fp16, greedy, temperature 0 without fallback, language es |
| Deepgram Nova-3 (es) | commercial API | commercial API | `deepgram_nova3_es` | streaming API (EU), language es, smart_format and punctuate, audio at 1x |
| Canary-1B-v2 | open weights | CC-BY-4.0 | `canary_1b_v2` | NeMo, greedy, source and target language es, punctuation on |
| Gemma 4 E4B | open weights | Apache-2.0 | – | bf16, model card ASR prompt with the language set to Spanish, greedy |
| Parakeet-TDT v3 | open weights | CC-BY-4.0 | `parakeet_tdt_0p6b_v3` | NeMo, greedy TDT decoding; language cannot be forced |
| Voxtral Mini 4B Realtime | open weights | Apache-2.0 | `voxtral_mini_4b_realtime` | bf16, whole clip offline, 480 ms delay, greedy; language cannot be forced |
| Hojo-ASR-Multi-V1 | open weights | Apache-2.0 | – | fp16, model card recipe (beam 4, repetition penalty 2.0); language cannot be forced |
| Whisper large-v3 LoS | open weights | Apache-2.0 | – | fp16, greedy, no timestamps, language es |
| omniASR-LLM-7B | open weights | Apache-2.0 | – | bf16, fairseq2 reference pipeline, v1 weights, language es |
| Granite Speech 4.1 2B NAR | open weights | Apache-2.0 | – | bf16, single non-autoregressive pass; language cannot be forced |
| MOSS-Transcribe-Diarize | open weights | Apache-2.0 | – | bf16, greedy, speaker tags removed from the output; language cannot be forced |
| Nemotron 3.5 ASR Streaming | open weights | OpenMDW-1.1 | – | 1.12 s streaming chunks, language es-ES, greedy |
| Deepgram Flux Multilingual | commercial API | commercial API | `deepgram_flux_multi` | streaming API (EU), language hint es, audio at 1x |
| Granite Speech 4.1 2B | open weights | Apache-2.0 | `granite_speech_4p1_2b` | bf16, model card ASR prompt, greedy; language cannot be forced |
| Whisper large-v3 clinical-assistance | open weights | Apache-2.0 | – | fp16, greedy, no timestamps, Spanish forced (language es) |
| Canary-1B-flash | open weights | CC-BY-4.0 | – | NeMo, greedy, source and target language es, punctuation on |
| Phi-4-multimodal-instruct | open weights | MIT | – | bf16, model card ASR prompt, greedy; language cannot be forced |
| Granite Speech 4.1 2B Plus | open weights | Apache-2.0 | – | bf16, model card ASR prompt, greedy; language cannot be forced |
| Parakeet-RNNT 1.1B es (projecte-aina) | open weights | Apache-2.0 | – | NeMo, greedy RNNT decoding |
| Voxtral Mini 3B | open weights | Apache-2.0 | `voxtral_mini_3b` | bf16, transcription mode with language es, greedy |
| VibeVoice-ASR | open weights | MIT | – | bf16, transformers port, greedy; language cannot be forced |

## Notes

* ¹ **OmniScribe 2 (Omniloy, self-hosted, not publicly available)**: Evaluated by Omniloy; not runnable with this package. OmniScribe 2 uses context from the patient's record. In this test that context included the medical terms spoken in each clinical clip, so its clinical numbers are a best case. It transcribed whole clips offline. Its real-speech rows used no context and are given rounded, without intervals.
* **Voxtral Small 24B (FP8 weights)**: Run with vLLM's FP8 weight-only quantization of the bf16 checkpoint (it fits in about 28 GB of GPU memory instead of about 55 GB); the bf16 model can score slightly differently.
* **Open-weights rows without a runner** (Gemma 4 E4B, Hojo-ASR-Multi-V1, Whisper large-v3 LoS, omniASR-LLM-7B, Granite Speech 4.1 2B NAR, MOSS-Transcribe-Diarize, Nemotron 3.5 ASR Streaming, Whisper large-v3 clinical-assistance, Canary-1B-flash, Phi-4-multimodal-instruct, Granite Speech 4.1 2B Plus, Parakeet-RNNT 1.1B es (projecte-aina), VibeVoice-ASR): not runnable with this package yet. Omniloy ran them outside the package in their default configuration (as defined at the top), on the same audio and with the same scoring; the *Systems* table gives the settings of each.
* **Type**: *commercial API* = hosted service called through its public streaming API; *open weights* = model run locally; *results only* = evaluated by its owner, not runnable with this package.
* **Options not used**: the commercial APIs offer custom vocabulary, keyterm or context features (Soniox, Deepgram) and Whisper accepts a text prompt. None of them was used; they could raise those systems' clinical scores.
* **`--model`**: the `fonendo run --model` name of the runner for the row's model; `–` means the row has no runner in this package yet (see *Open-weights rows without a runner* above; results-only rows are not runnable). *(experimental)*: the runner was not re-run with this package against the published row, so the reproduction is not verified (`fonendo models` lists these runners).
* **Degenerate**: share of clips whose output is empty, loops or runs away; such outputs are scored as they are (an empty output counts every reference word as deleted). Metrics are corpus-level, so one runaway output of hundreds of words can dominate a system's WER and insertion rate; a very wide interval is the sign of it.
* Metric definitions and the bootstrap procedure: see the README, section *Methodology*.
