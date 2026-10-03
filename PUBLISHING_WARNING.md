# Publishing warning — review before making the course public

Light review needed. The course is legitimate research-engineering material, but two modules
touch dual-use alignment and safety research.

What to check before publishing:

- **Module 18 (alignment science, frontier evaluation and governance)** and **lesson 16.4 (reward hacking)**.
  - Labs reproduce published model-organism results (emergent misalignment from narrow
    fine-tuning, reward hacking in a planted verifier) only with benign proxy behaviours, such as
    insecure-looking toy code or sycophancy.
  - Check that no lab produces, or explains how to produce, a model with harmful capability, and
    that no lab shows how to remove a released model's safety training.
  - Check that the misalignment discussion stays at the level of the cited papers.
- **lesson 17.4 (steering)**: steering examples use harmless traits only (style, sycophancy, topic).
  No refusal-direction removal (that is the separate `abliteration` course and its own warning).
- **lessons 18.3–18.4 (evaluation and safety frameworks)**: dangerous-capability evals are described from the safety frameworks
  and system cards. No hazardous task content (bio, chem, cyber exploitation) is reproduced.

If all of the above holds, the course needs no risk notice to learners beyond its GPU-cost note.
