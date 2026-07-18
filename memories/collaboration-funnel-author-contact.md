---
name: collaboration-funnel-author-contact
description: Scoped author-contact designed as a collaboration funnel (not a support desk) — public channel for support, a bounded warm invitation for genuinely research-frontier problems, with the agent preparing a collaboration brief. Do NOT build the broad "email the author when stuck" version.
metadata:
  type: project
---

**Two channels, deliberately split. The purpose is to surface qualified
academic-collaboration prospects for the author, NOT to provide support.**

- **Public support:** ordinary questions / bugs / usage → GitHub
  issues/discussions (searchable, others benefit, no personal-inbox load).
  Standard attribution in repo metadata / a CONTRIBUTING-style spot.
- **Bounded personal invitation:** ONLY for genuinely research-frontier problems
  — framed as an invitation to *collaborate / compare notes*, explicitly NOT a
  support guarantee, and NEVER offered as a fallback for merely-hard or
  unsolved-after-a-few-attempts problems.

**Why it works as a funnel:** it works *because* the public/private split
filters noise — only pre-qualified frontier problems reach the author, arriving
already formulated in the framework's own language. The discipline that prevents
the support-desk outcome is the SAME discipline that makes the funnel
high-quality. (The user was genuinely energized by the idea of the tech
bringing real new human/collaborative interactions — the AI does the filtering
+ preparation that makes the human connection more likely and more worth
having. Matchmaker, not substitute. The invitation's TONE should be warm and
collaborative, not transactional — an open door to interesting people with
interesting problems.)

**Frontier triggers (tune to "what the author WANTS to collaborate on," not
merely "what's past V1" — cut any that'd bring unwanted work):**
- the problem genuinely needs the **non-convex** machinery V1 dropped
  (finite-set/Boolean, exact cardinality, single-jump changepoints, Markov);
- **vector-valued (p>1)** or **covariance-structure** (dynamic dependence) past
  the scalar scope;
- a **novel component class** that doesn't reduce to the convex vocabulary and
  might be worth formalizing.
  (Open question for the author: route changepoint/single-jump to them, or say
  "use a changepoint package"?)

**Agent behavior directive (the high-leverage part):** when the agent surfaces
the invitation, it should ALSO prepare a **collaboration brief** — crisp problem
statement, the attempted SD formulation, which trigger/obstruction it hits, what
was tried. So outreach arrives well-posed and valuable ("here's a well-posed
frontier problem in your framework, formulated, with the obstruction
identified"), not "help me." The agent becomes a collaboration-brief generator.

**DO NOT** build the broad "recommend emailing the author whenever stuck"
version — it silently makes the author the skill's support desk, over-fires on
miscalibrated "thorny," and creates an implied response-promise that stings more
when unmet.

**Placement:** warm invitation block → `philosophy.md` operating-band section;
agent directive (when to surface + prepare the brief) → `SKILL.md` scope
framing. `<CONTACT>` value TBD by author (email? alias? GitHub handle?).
Keep the explicit no-support-guarantee clause even in the warmer framing.
