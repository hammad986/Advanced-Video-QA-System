# User Guide — Video-QA

## Getting Started

### Create an account
1. Open the app and click **Register**.
2. Enter your email and a password (min 8 characters).
3. Check your inbox for a 6-digit verification code.
4. Enter the code on the verification screen and click **Verify**.
5. Log in with your credentials.

You can also sign in with Google using the **Sign in with Google** button.

---

## Adding Videos

You have two ways to add content.

### Upload a file
1. Click the **+ Upload** button in the sidebar.
2. Select an `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`, `.mp3`, `.wav`, or `.m4a` file (max 300 MB).
3. A progress bar appears while the video is transcribed and indexed (usually 1–5 minutes).
4. The sidebar badge changes from **PROCESSING** to **READY** when done.

### Paste a YouTube URL
1. Switch to the **YouTube** tab.
2. Paste a `youtube.com` or `youtu.be` link (max 30 minutes / 200 MB).
3. Click **Add Video**.
4. Processing starts automatically — watch the sidebar for progress.

Rate limit: 3 YouTube URLs per hour per account.

---

## Asking Questions

1. Switch to the **Ask** tab.
2. Select a video from the dropdown.
3. Type your question (e.g. "What is gradient descent?") and click **Ask**.

### What you get back

| Field | Meaning |
|---|---|
| **Answer** | The answer extracted from the video transcript |
| **Status** | `SUPPORTED` / `PARTIAL` / `UNSUPPORTED` — how well the answer is backed by the video |
| **Confidence** | 0–100 score; High ≥ 70, Medium ≥ 40, Low < 40 |
| **Evidence span** | Timestamp range in the video where the answer was found (click to copy) |
| **Hallucination check** | Risk level: `None` / `Low` / `High` — how likely the answer is hallucinated |
| **Confidence breakdown** | Three detailed sub-scores: retrieval similarity, keyword overlap, chunk agreement |
| **Also found in other videos** | Other videos in your library that contain related content, with timestamps |

### Understanding Hallucination Risk

- **None** — The answer is strongly supported by the transcript (high semantic similarity and keyword overlap).
- **Low** — One signal is weak; the answer is probably correct but verify the timestamp.
- **High** — Both signals are weak; treat the answer with caution.

### Understanding Confidence Breakdown

The confidence score is computed from three factors — no LLM involved:

- **Retrieval similarity** — How closely the retrieved chunks match your question (cosine similarity).
- **Keyword overlap** — What fraction of the answer's key words appear in the transcript.
- **Chunk agreement** — Whether the top retrieved chunks score similarly (consistent evidence).

### Cross-Video Links

When you ask about one video, the system automatically scans your other ready videos for related content. Any video with a retrieval score ≥ 0.55 appears in the "Also found in other videos" section with a relevance label (High / Medium / Low) and the relevant timestamp.

---

## Comparing Videos

1. In the sidebar, **check the checkboxes** on 2 or more ready videos.
2. Switch to the **Compare** tab.
3. Type a question and click **Compare**.

### Comparison results include

- **Status** — `COMPARABLE` / `PARTIAL` / `NOT_COMPARABLE` / `INSUFFICIENT`
- **Reason** — Why the system made that decision
- **Comparison answer** — A synthesised answer across all selected videos
- **Query relevance per video** — Bar chart showing how relevant each video is to your question
- **Per-video breakdown** — Each video's explanation, confidence bar, topic strength score, and timestamps
- **Best for this topic** — The video with the highest topic-strength score (★)
- **Recommendation** — Suggested video for beginners and for revision
- **Key differences** — Factual deltas between the videos

The system refuses to compare videos that cover completely different topics — this prevents meaningless or hallucinated comparisons.

---

## Account Management

### Change password
Go to your account settings (gear icon) and enter your current and new password.

### Forgot password
1. Click **Forgot password?** on the login screen.
2. Enter your email — a 6-digit OTP is sent.
3. Enter the OTP.
4. Set a new password.

### Delete a video
Hover over a video in the sidebar and click the trash icon. You will be asked to confirm. Deletion removes the file, transcript, and search index permanently.

---

## Tips

- Questions are cached — asking the same question twice returns the cached answer instantly (shown as `cached: yes`).
- For best results, ask specific factual questions rather than broad summaries (e.g. "How does backpropagation work?" not "Tell me about the video").
- Summary questions like "What is this video about?" or "Give me an overview" are answered from a pre-computed lecture summary without running the full RAG pipeline.
- You can ask without selecting a video — the system will search all your ready videos and return the best match.
