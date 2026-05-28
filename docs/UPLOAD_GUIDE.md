# Uploading Documents & Testing the AI

A step-by-step guide for adding documents (cases, memos, statutes) to
Legalite AI and confirming the AI has actually learned from them.

You do not need to use a terminal. Everything in this guide happens in
your web browser.

---

## 1. Open the upload page

Open this URL in Chrome, Edge, or Safari:

```
https://<your-legalite-ai-url>/docs
```

(Ask your engineer for the exact URL if you don't have it. On a
development machine it's usually `http://localhost:8000/docs`.)

You'll see a page titled **Legalite AI** with a long list of expandable
sections. This is the API explorer — it lets you upload files and ask
questions through a web form.

---

## 2. Upload a document

1. Scroll down until you see a green box labelled
   **`POST  /upload-law`**. Click it to expand.
2. Click the white **"Try it out"** button on the right.
3. Fill in the three fields:

   | Field      | What to put                                                                                                                                                 |
   | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `file`     | Click **Choose File** and select a PDF from your computer. PDFs only.                                                                                       |
   | `title`    | A short, human-readable name for the document. Example: `"Republic v. Mensah (2018)"` or `"Sample Memo — Land Dispute Tema"`.                              |
   | `doc_type` | One of: `Constitution`, `Act`, `Regulation`, `Case Law`, `Memo`, `Sample Case`, `Template`. Pick the one that best describes the document. Case-sensitive. |

4. Click the blue **"Execute"** button.

### What happens next

The page will show one of three things below the button:

- **Code 201** — green box. Upload succeeded. The document is now in
  the AI's knowledge base. The response will include a
  `document_id` (a long string of letters and numbers) and a
  `chunks_indexed` count (how many searchable pieces the PDF was
  broken into).
- **Code 400 / 413 / 422** — yellow box. Something is wrong with the
  file or the form. Common reasons:
  - File is not a PDF.
  - File is larger than 50 MB.
  - `title` is too short.
- **Code 500** — red box. The server tried but failed. Send the full
  error message to your engineer.

### Maximum file size

PDFs must be **50 MB or smaller**. If a case file is larger than this,
either ask your engineer to split it, or split it yourself using a free
tool like Adobe Acrobat's "Split PDF" feature.

### How long does it take?

- Short PDFs (a few pages): about 5–10 seconds.
- Long PDFs (a full Act, the Constitution): 1–3 minutes. Be patient
  and don't click Execute twice — the second click would create a
  duplicate.

---

## 3. Confirm the AI actually learned from the document

Uploading a document is not enough. You should **always test** that the
AI can find and use it. Here's how.

### Step A — Think of a question only this document can answer

Open the PDF you just uploaded and find a specific, narrow fact.
Examples:

- For a case: _"What was the holding in Republic v. Mensah?"_
- For an Act: _"What does Section 12 of the Land Act say about
  customary tenure?"_
- For a memo: _"What did our office advise on the Tema land dispute?"_

Avoid vague questions like _"Tell me about land law"_ — they don't
test whether **this specific document** was learned.

### Step B — Ask the question via the API explorer

1. Back on the `/docs` page, scroll to **`POST  /ask`**. Click to
   expand it.
2. Click **"Try it out"**.
3. In the **Request body** box, replace the placeholder with:

   ```json
   {
     "question": "What was the holding in Republic v. Mensah?"
   }
   ```

   Use the exact question you thought of in Step A. Make sure you only
   change the text inside the quotes — keep the curly braces and the
   word `"question"` exactly as shown.

4. Click **"Execute"**.

### Step C — Read the answer carefully

The response is a block of structured information. Four fields tell you
whether the AI actually learned from your document:

| Field             | What to look for                                                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `answer`          | A real legal answer that references facts from your PDF. NOT _"I could not confidently determine this…"_                                                       |
| `confidence`      | `"high"` or `"medium"`. **If it says `"low"`, the AI did NOT use your document confidently.** Investigate further.                                            |
| `citations`       | A list of references. Each one points to a section, article, or page of a source document. Your uploaded document's title should appear here.                |
| `sources_used`    | A list of document titles. Your uploaded document's `title` (from Step 2) should appear in this list.                                                         |

### What "good" looks like

```json
{
  "answer": "In Republic v. Mensah (2018), the Supreme Court held that…",
  "confidence": "medium",
  "citations": [
    { "law": "Republic v. Mensah (2018)", "page_start": 7, "page_end": 8 }
  ],
  "sources_used": ["Republic v. Mensah (2018)"],
  ...
}
```

The title you typed in Step 2 appears in `sources_used`. The AI is
quoting from the right place. **You're done.**

### What "bad" looks like

```json
{
  "answer": "I could not confidently determine this from the provided Ghana legal materials.",
  "confidence": "low",
  "citations": [],
  "sources_used": [],
  "reasoning_summary": "top retrieval score 0.18 below threshold 0.35"
}
```

If you see this for a question the PDF **clearly answers**, something
went wrong with the upload. See "Troubleshooting" below.

---

## 4. Troubleshooting

### "The AI says it doesn't know, but I just uploaded the document!"

1. **Wait 30 seconds and try again.** Indexing isn't always instant.
2. **Check your question.** Use words that actually appear in the
   document. The AI matches semantically, but very vague or unrelated
   wording can miss.
3. **Confirm the upload succeeded.** Go back to your upload response and
   make sure `chunks_indexed` is **greater than 0**. If it's 0, the
   PDF was empty or unreadable (e.g. a scan of a photocopy).
4. **Try a different question.** Pick a different specific fact from
   the same document. If the AI knows that one but not the first,
   it's a question-wording issue, not an upload issue.
5. **If none of the above works**, send your engineer:
   - The PDF you uploaded
   - The `document_id` from the upload response
   - The exact question and the full response

### "I uploaded the same document twice by accident"

This is mostly harmless — both copies are searched, the AI just sees
two of every passage. Ask your engineer to delete the duplicate when
convenient. Provide both `document_id` values if you can find them.

### "The upload took forever and timed out"

Large PDFs (40+ MB, or scanned documents requiring OCR) can take
several minutes. If the browser shows a timeout, **don't re-upload
immediately** — the document might still be ingesting in the
background. Wait 5 minutes, then test with a question (Step 3). If the
AI still can't find it after 5 minutes, then it's safe to retry.

---

## 5. Best practices for building the corpus

- **Use clear, consistent titles.** _"Republic v. Mensah (2018)"_ is
  better than _"mensah case scanned final v2.pdf"_. The title is what
  shows up when the AI cites the source.
- **Pick the right `doc_type`.** It affects how the AI weighs the
  source: a Constitution carries more weight than a Memo.
- **Upload digital PDFs, not scans, when possible.** Scanned documents
  go through OCR, which is slower and sometimes misreads characters.
- **Test every important document** with at least one question
  (Section 3) before moving on. Catching a failed upload immediately
  is much easier than discovering it weeks later.
- **Spread heavy uploads out.** If you have hundreds of files, upload
  them in batches over the course of a day rather than firing them
  all at once. The AI will index them as they arrive.

---

## Reference card

```
Upload:   POST /upload-law      (title, doc_type, file)
Test:     POST /ask             ({"question": "..."})
Explorer: https://<host>/docs
```

Questions or problems → contact your engineer with the `document_id`
and a screenshot of the response.
