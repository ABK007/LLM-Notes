# Concepts in GenAI Foundational Large Language Models Whitepaper - 2k25

### What is Tree of thought?

ToT stands for "Tree of Thought." It’s a way to solve tricky problems that need you to try out different ideas or possibilities—something we call "exploration." Imagine you’re figuring out a puzzle with lots of steps, and you need to test different ways to get to the answer. ToT helps with that!

### How Does It Work?

ToT organizes your thinking like a tree (think of a family tree or a branching diagram). Here’s the key idea:

- **Thoughts**: Each "thought" is a short, clear sentence or idea (called a "coherent language sequence") that’s like a stepping stone toward solving the problem.
- **Tree Structure**: These thoughts are connected in a tree. You start with one main thought (the root), and then it branches out into more thoughts, which can branch out even further.
- **Exploring Paths**: The "tree" lets you try different directions. From each thought, you can branch off into new ideas, creating multiple paths to explore. It’s like choosing different routes on a map to reach your destination.

This setup is great for complex tasks because it lets you test lots of options in an organized way.

### Where Did It Come From?

There’s a notebook and a paper called _‘Large Language Model Guided Tree-of-Thought’_ that explain ToT in more detail. Basically, it’s a method that uses big AI language models (like the ones that can chat or write text) to guide this tree-building process. The "ReAct" part seems incomplete in your text, so we’ll skip that since we don’t have enough info.

### Example to Make It Simple

Let’s say you’re planning a trip from New York to Los Angeles, but you want to keep costs low and maybe visit a cool city on the way. Here’s how ToT could help:

1. **Starting Thought (Root)**: "I need to get from New York to Los Angeles cheaply."

   - This is where the tree begins.

2. **Branching Out (First Level)**:

   - **Thought 1**: "Take a direct flight to Los Angeles."
   - **Thought 2**: "Travel by bus through Chicago."
   - **Thought 3**: "Drive with a friend and split gas costs."

3. **Branching Further (Second Level)**:
   - From **Thought 2** ("Travel by bus through Chicago"):
     - **Thought 2.1**: "Stay a day in Chicago to sightsee, then continue by bus."
     - **Thought 2.2**: "Take the fastest bus route with no stops."
   - From **Thought 3** ("Drive with a friend"):
     - **Thought 3.1**: "Stop in Denver to visit a friend."
     - **Thought 3.2**: "Drive straight through to save time."

Each thought is a step toward solving your problem (a cheap trip). The tree lets you explore all these options: flying, bussing, driving, stopping, or not stopping. You can then pick the best path—like maybe "Thought 2.2" if it’s the cheapest and fastest.

### Why Is This Cool?

ToT is awesome for hard problems because it:

- Breaks them into smaller, manageable pieces (thoughts).
- Lets you try lots of ideas without getting lost.
- Helps you find the best solution by comparing all the paths.

So, in short, the Tree of Thought is like a map for your brain (or an AI) to explore different ways to solve something tricky.

---

## How Decoder works:

### What’s Happening Here?

This line is talking about the **decoder**, which is one half of a Transformer model (the other half is the **encoder**). The encoder’s job is to take the input—like a sentence in French—and turn it into a kind of "summary" or **representation** that captures its meaning. The decoder then takes that summary and uses it to create the output—like translating that French sentence into English.

### What Does "Representation" Mean?

The "representation" is like a compressed version of the input sentence that the Transformer understands. It’s not the actual words, but a mathematical form of the sentence’s meaning, created by the encoder. Think of it as a recipe card: it doesn’t list every step of cooking, but it has all the key info needed to make the dish.

#### Example:

- Input (French): _"Le chat est noir."_ ("The cat is black.")
- The encoder turns this into a representation—a bundle of numbers that captures the meaning (something like "there’s a cat, and it’s black").
- The decoder gets this representation and uses it to build the English version.

### What Does "Autoregressively" Mean?

"Autoregressively" is a fancy way of saying the decoder generates the output **one word at a time**, using what it’s already written to figure out the next word. It’s like writing a sentence where each word you pick depends on the words you’ve chosen so far.

#### How It Works:

- The decoder doesn’t spit out the whole translation at once. Instead, it builds it step by step.
- It starts with a blank slate (or a special starting signal), looks at the representation from the encoder, and predicts the first word.
- Then it takes that first word, combines it with the representation, and predicts the second word.
- It keeps going like this until it’s done.

#### Example in Action:

Let’s say the input is _"Le chat est noir."_ and the decoder is translating it to English:

1. **Step 1**: The decoder looks at the representation and guesses the first word: _"The"_.
2. **Step 2**: It takes _"The"_ plus the representation and predicts the next word: _"cat"_.
3. **Step 3**: With _"The cat"_ and the representation, it predicts _"is"_.
4. **Step 4**: With _"The cat is"_, it predicts _"black"_.
5. **Step 5**: It decides the sentence is complete (often signaled by a special "end" token).

So, the output becomes _"The cat is black."_—built one word at a time, with each new word depending on what came before.

### Why "Autoregressively"?

The "auto" part means "self," and "regressive" means looking back. The decoder is "self-looking-back" because it uses its own previous outputs (the words it’s already generated) to decide what comes next. This makes the translation smooth and natural, because each word fits with the ones before it.

### Tying It All Together

So, the line means: The decoder takes the meaning-packed representation from the encoder and uses it to write the output text (like an English translation) one word at a time, where each new word builds on the ones it’s already made. It’s like a translator who listens to a French sentence, holds the idea in their mind, and then carefully speaks it in English, word by word, making sure it all flows.

---

## How DeepSeek Trained its Models?

DeepSeek, a company working on smart AI language models, came up with a new way to teach their model, called **DeepSeek-R1-Zero**, to think and reason really well—like solving tricky math problems. They wanted it to be as good as OpenAI’s “o1” series (a super-smart AI), but without needing a huge pile of examples (called labeled data) to train it. And guess what? They did it! Their model matched “o1” on a tough math test called **AIME 2024**, proving it can handle complex reasoning.

They used something called **reinforcement learning (RL)**, which is like training an AI by rewarding it for good answers. But instead of following the usual RL playbook, they invented a fresh trick called **Group Relative Policy Optimization (GRPO)**. This method skips a part that most RL systems need—a “critic” model—and still gets amazing results. Let’s unpack this step by step.

---

## What’s Reinforcement Learning (RL)?

Imagine you’re teaching a kid to ride a bike. You don’t give them a rulebook; instead, you cheer when they pedal without falling and help them up when they crash. Over time, they figure out how to balance by trying again and again. That’s RL: learning through trial and error with rewards for doing well.

For an AI language model, RL works the same way. The model tries to answer a question—like “What’s 5 + 3?”—and gets a “reward” if it’s right (e.g., “8” gets a high score, “7” gets a low one). The more it practices, the better it gets at giving correct and helpful answers.

---

## The Old Way: RL with a “Critic” Model

In traditional RL for language models, there’s a helper called a **critic model**. Think of it as a teacher who’s already been trained on tons of examples—like thousands of math problems with correct answers. The main model guesses an answer, and the critic checks it, saying, “Nope, that’s wrong” or “Nice job!” This feedback helps the model improve.

Here’s an example:

- The model says: “5 + 3 = 6”
- The critic, trained on labeled data, says: “Wrong! It’s 8.”
- The model learns from the correction.

But there’s a problem: training the critic takes a lot of labeled data (e.g., solved math problems or perfect essays), which is hard to gather. It’s like needing a giant answer key before you can even start teaching.

---

## DeepSeek’s New Way: GRPO (No Critic Needed!)

DeepSeek said, “Forget the critic!” With their **GRPO** method, the model teaches itself without a teacher. Here’s how it works:

1. **Generate Multiple Answers**: The model gets a question, like “What’s 5 + 3?” and comes up with several guesses—say, “6,” “8,” and “9.”
2. **Score Them with Rules**: Instead of a critic, DeepSeek uses simple rules to grade each answer. These rules check things like:
   - **Coherence**: Does the answer make sense?
   - **Completeness**: Did it fully solve the problem?
   - **Fluency**: Is it clear and natural (if it’s text)?
     For our math example, “8” would score high because it’s correct, while “6” and “9” would score lower.
3. **Compare to the Group**: The model looks at all its answers and sees how each one stacks up against the average. If “8” is better than the others, it learns to lean toward that kind of answer next time.

It’s like a kid writing three short stories, then using a checklist (e.g., “Does it have an ending?”) to pick the best one and learn from it—no teacher required!

---

## Example in Action

Let’s say the task is to answer: “Why do birds fly?”

- **Traditional RL with a Critic**:

  - Model says: “Birds fly because they like it.”
  - Critic (trained on good answers) says: “Not quite. They fly to find food and escape danger.”
  - Model adjusts based on the critic’s feedback.

- **DeepSeek’s GRPO**:
  - Model tries three answers:
    1. “Birds fly because they like it.”
    2. “Birds fly to find food and avoid predators.”
    3. “Birds fly because the sky is blue.”
  - Rules score them:
    - Answer 1: Low (not very complete or accurate).
    - Answer 2: High (coherent, complete, and clear).
    - Answer 3: Low (doesn’t make sense).
  - The model sees Answer 2 is the winner and learns to give answers like that.

This “self-play” lets the model improve without needing anyone to label what’s right or wrong.

---

## Why This Is Awesome

1. **No Need for Tons of Data**: Labeled data is expensive and slow to make. GRPO skips it, saving time and effort.
2. **Self-Learning Power**: The model gets better by challenging itself, like a chess player practicing against a mirror.
3. **Top-Notch Results**: DeepSeek-R1-Zero scored as well as OpenAI’s “o1” on the AIME 2024 math test, which is a big deal for a model trained this way.

---

## But There Were Some Hiccups

At first, DeepSeek’s pure RL approach had issues:

- **Hard-to-Read Answers**: The model’s outputs were sometimes confusing or messy, like a genius scribbling notes no one else could follow.
- **Language Mixing**: It might start in English, then toss in random words from another language, making it hard to understand.

Imagine asking, “What’s 2 + 2?” and getting: “Four, muy bueno, yes!” It’s correct but weird. DeepSeek later tweaked their process to fix these problems, but it shows RL alone isn’t perfect—it needs polishing.

---

## Why It Matters

DeepSeek’s GRPO method is a game-changer because it proves AI can get super smart at reasoning—like solving math or answering tough questions—without relying on mountains of human-made examples. It’s like teaching a student to learn by experimenting instead of memorizing a textbook. This could make building powerful AI faster and cheaper in the future.

Still, the early hiccups remind us that while this approach is brilliant for reasoning, it needs extra work to make the answers clear and friendly for humans. DeepSeek pulled it off, and their success with DeepSeek-R1-Zero shows how exciting this new path could be!

---

