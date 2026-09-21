# LegaLite review guide (for the legal reviewer)

You don't need to understand any of the software behind LegaLite to do
this. You don't need to know what a database, a vector, or an API is. This
document explains what we need from you and why.

## What we need

About **15-20 real legal questions**, each with the correct answer,
written the way you'd naturally answer them. That's it.

For each question, please provide:

```
Question:

Category:            (see the list below)
Difficulty:           Easy / Medium / Hard

Expected answer:
   (write the correct answer in your own words)

Expected legal authority/citation(s):
   (the Act, section, case, or provision that supports the answer)

Expected document(s):
   (which specific law(s) or case(s) should this answer draw from)

Expected passage(s):
   (optional — a short quote from the source that supports the answer,
   if you have one handy)

Should LegaLite refuse to answer?
   Yes / No
   (Yes = the question is genuinely outside what LegaLite's documents
   cover, or too ambiguous to answer safely)

Verified:
   Yes / No
   (Yes only once you're confident the above is actually correct — see
   "What 'verified' means" below)
```

## What's the difference between these fields?

They can look like the same thing at first glance, so here's what makes
each one distinct:

- **Expected answer** — what a legally correct response should actually
  *establish* or conclude. This is the "so what" — write it the way you'd
  explain it to the client asking the question.
- **Expected citation(s)** — the specific authority (an Act and section, a
  case name) that *supports* the answer. This is "what proves it's true."
  A question can need more than one — see multi-document questions below.
- **Expected document** — which whole law or case the answer should come
  from, by name. Broader than a citation: you might know the right
  *document* without knowing the exact section number.
- **Expected passage** — the actual sentence or two, quoted from the
  source, that backs up the answer. Narrower than a citation: this is the
  literal text, not just "which section." Optional — skip it if you don't
  have the source text in front of you.
- **Expected refusal** — whether the *correct* behavior is for LegaLite to
  say "I don't have enough information," rather than attempt an answer at
  all. This is a yes/no, and it's usually "no" — most questions should get
  a real answer.

If a question doesn't need one of these (e.g. no exact passage on hand),
leave it blank. Partial information is still useful.

## Why this matters

We want to measure whether LegaLite is actually getting more accurate over
time — not just whether it *sounds* confident. The only way to measure that
honestly is to compare its answers against questions where we already know
the right answer, confirmed by someone with legal training. Without your
input, every number LegaLite's engineering team looks at is a guess dressed
up as a measurement.

## What "verified" means

Mark a question `Verified: Yes` only once you've personally confirmed:

- The expected answer is legally accurate.
- The authority/citation you gave actually supports that answer.
- The document or case you named is one LegaLite is actually meant to have
  (if you're not sure what's in its library, ask the engineering team —
  don't guess).
- If you said LegaLite should refuse, that's genuinely the right call.

If you're not fully sure about any of these, mark it `Verified: No` for
now — a question can still be useful un-verified (the team can look at how
LegaLite handles it), it just won't count toward the trustworthy number.

## Categories

Pick whichever fits best — don't overthink it:

- **Direct statutory** — "what does the law say about X" (aim for ~4)
- **Legal concept** — explaining a legal concept, not one specific
  provision (aim for ~3)
- **Case-law** — questions needing a specific court decision (aim for ~3)
- **Multi-document** — the answer genuinely needs more than one source
  (a statute AND a case, for instance) (aim for ~3)
- **Citation/source** — the main point is identifying the right authority
  to cite (aim for ~2)
- **Current/recent law** — only if you know of a recent amendment or
  decision that changed something (aim for ~2, skip if nothing applies)
- **Should refuse** — a question LegaLite genuinely shouldn't be able to
  answer from its documents, or that's too vague to answer safely (aim
  for ~2-3)

These are just targets to get a good spread — don't feel boxed in if your
questions naturally land differently.

## What happens with your answers

Your answers go into a file the engineering team uses to run LegaLite's
questions automatically and compare its actual answers against yours. You
will never need to open that file, write any code, or use any development
tools — just send your answers back in whatever format is easiest for you
(a document, a spreadsheet, an email), and the team will do the rest.
