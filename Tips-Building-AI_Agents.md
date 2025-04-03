# Tips For Building Affective Agents - Anthropic

## 1. Overview

The video features a roundtable discussion among experts (Alex, Erik, and Barry) who share insights from a recent blog post titled _Building Effective Agents_. The conversation revolves around what an “agent” is in the context of AI, how it differs from a fixed “workflow,” and what practical implications these differences have for developers. The speakers also offer opinions on the current hype around consumer-facing agents.

## 2. Key Concepts and Definitions

### Agent

- **Definition:**  
  An agent is an AI system that is given autonomy to decide how many actions or steps to take in order to resolve a task. Instead of following a fixed sequence, it “loops”—continuously making decisions and taking actions (e.g., web searches, code edits) until it reaches a solution.
- **Characteristics:**
  - **Autonomous Decision-Making:** The model chooses its next action without a predefined path.
  - **Iterative Process:** It can decide to call its underlying language model (LLM) multiple times based on the evolving context or output.
  - **Dynamic Resolution:** There is uncertainty in the number of steps required, making it adaptable to varied problem complexities.
- **Example Use Cases:**
  - Handling customer support queries where the solution path isn’t fixed.
  - Iterating over code changes until the desired output is achieved.
    > “…what we think an agent is is where you're letting the LLM decide sort of how many times to run… until it's found a resolution.” citeturn0file0

### Workflow

- **Definition:**  
  A workflow is a series of predefined, linear steps where each LLM call or prompt has a specific role. The sequence is predetermined and “on rails,” meaning the developer controls the entire process from start to finish.
- **Characteristics:**
  - **Fixed Sequence:** A clear, linear progression from one prompt to the next.
  - **Predictability:** The outcome and the number of steps are known beforehand.
  - **Simplicity:** Each prompt transforms one input into one output in a predictable manner.
- **Example Use Cases:**
  - Simple categorization tasks where the answer is refined through a series of fixed transformations (e.g., categorizing a user question into a set number of categories).
    > “…a workflow prompt looks like, you have one prompt… take the output, feed it into prompt B… and then you’re done. Kind of, there’s this straight line fixed number of steps.” citeturn0file0

### LLM (Large Language Model)

- **Definition:**  
  An LLM is an AI model trained on vast amounts of textual data that can generate human-like responses. In this context, it is used as the core engine behind both agents and workflows.
- **Role in Agents vs. Workflows:**
  - In workflows, LLMs are invoked in a controlled, sequential manner.
  - In agents, the LLM is given more freedom to decide how many times to run and which actions to take.

### Prompt

- **Definition:**  
  A prompt is the input provided to an LLM to elicit a response.
- **In Context:**
  - **Workflow Prompt:** Typically designed for a single transformation step.
  - **Agent Prompt:** More open-ended, often includes instructions or “tools” that allow the LLM to perform multiple actions (e.g., searching the web, editing code).

### Tools and Code Orchestration

- **Tools:**  
  Agents can be equipped with various tools (e.g., web search, code execution, file editing) that extend their capabilities beyond a single LLM call.
- **Code Orchestration:**  
   This involves combining multiple LLM outputs and external tool interactions into a coherent process—allowing an agent to iterate until the final output meets the requirements.
  > “…an agent prompt will be sort of much more open-ended and usually give the model tools or multiple things to check… run code and keep doing this until you have the answer.” citeturn0file0

## 3. Detailed Discussion Points from the Video

### The Hype around Consumer Agents

- **Observation:**  
  One of the speakers remarks that consumer-facing agents are “over-hyped.”
- **Context:**  
   The point is made that while the idea of having an agent fully automate tasks like booking a vacation is appealing, in practice it can be nearly as challenging as performing the task manually.
  > “Trying to have an agent fully book a vacation for you, almost just as hard as just going and booking it yourself.” citeturn0file0

### Differentiating Agents from Workflows

- **Multiple Definitions:**  
  There’s recognition that many definitions exist for “agent.” The speakers stress that it’s important to distinguish between a simple chain of LLM calls (workflow) and a truly autonomous agent.
- **Autonomy vs. Predefined Steps:**
  - **Agent:** Lets the LLM decide “how many times to run” and when to stop.
  - **Workflow:** Has a predetermined series of steps.
    > “It’s more autonomous, whereas a workflow you can kind of think of as… on rails through a fixed number of steps.” citeturn0file0

### Evolution of AI Systems

- **Historical Perspective:**  
   Initially, many systems used a single LLM call. As models improved and teams became more sophisticated, the architecture evolved into using multiple LLMs that could orchestrate themselves—prompting the need to distinguish agents from workflows.
  > “…we both worked with a large number of customers who are very sophisticated… and we kind of went from having a single LLM to having a lot of LLMs and like eventually having LLMs orchestrating themselves.” citeturn0file0

### Practical Implementation: Prompts and Code

- **Workflow Implementation:**  
  A developer writing a workflow knows exactly how many prompts will be used and in what order. Each prompt is tightly controlled, often with intermediate checks.
- **Agent Implementation:**  
   An agent prompt is designed to be more exploratory. It includes multiple “tools” (for example, instructions for web searching or code editing) so the model can iterate and determine its own path to the solution.
  > “…in contrast, an agent prompt will be sort of much more open-ended… giving the model tools… or run code and keep doing this until you have the answer.” citeturn0file0

## 4. Practical Takeaways for Developers

- **When to Use a Workflow:**

  - For tasks with clear, fixed steps.
  - When predictability and simplicity are desired.
  - Example: A simple categorization pipeline where the input is transformed in a known sequence.

- **When to Use an Agent:**

  - For tasks where the number of required steps is uncertain.
  - When tasks require adaptability, such as interacting with external tools (web search, code editing).
  - Example: Customer support interactions that may require back-and-forth communication and iterative problem solving.

- **Designing Agent Prompts:**

  - Provide clear instructions while leaving room for the model to decide on subsequent actions.
  - Include multiple “tools” or possible actions that the agent can take, ensuring it can loop until a satisfactory resolution is reached.

- **Designing Workflow Prompts:**
  - Keep each prompt simple and deterministic.
  - Ensure that the outputs at each stage are validated before moving to the next step.

## 5. Definitions of Key Terms in Context

- **Agent:** An autonomous AI system that determines its course of action dynamically, iterating through tasks until it finds a solution.
- **Workflow:** A sequence of pre-determined steps executed in a fixed order, typically used when the process is straightforward and predictable.
- **LLM (Large Language Model):** The underlying AI that generates responses; used in both agents and workflows.
- **Prompt:** The input given to an LLM that triggers a response; can be designed for either a fixed transformation (workflow) or an exploratory process (agent).
- **Tools:** External capabilities (e.g., web search, code execution) that can be integrated into agent prompts to extend the functionality beyond simple text generation.
- **Code Orchestration:** The process of managing multiple LLM calls and tool interactions to achieve a final, coherent outcome.

Below is a comprehensive documentation that covers the full set of subtitles from the video on tips for AI agents. This documentation not only explains the key discussion points and anecdotes from the experts but also defines the core terms and concepts mentioned throughout the conversation.

---

## 2. Core Definitions and Concepts

### Agent

- **Definition:**  
  An agent is an AI system that is given the freedom to decide its own sequence of actions. Instead of following a rigid set of steps, the agent is empowered to loop, iterate, and adapt until it finds a satisfactory resolution.
- **Key Characteristics:**
  - **Autonomy:** The agent decides how many times to run its underlying LLM (Large Language Model) calls.
  - **Iteration:** It continuously re-evaluates its next step (e.g., running a web search or editing code) based on intermediate outputs.
  - **Dynamic Resolution:** The exact number of steps isn’t predetermined, making the process flexible.
    > “…what we think an agent is is where you're letting the LLM decide sort of how many times to run… until it's found a resolution.” citeturn0file0

### Workflow

- **Definition:**  
  A workflow is a series of predefined, sequential steps where each prompt or LLM call has a specific role. The sequence is fixed and predictable.
- **Key Characteristics:**
  - **Fixed Sequence:** A clear, linear progression from one step to the next.
  - **Deterministic Behavior:** Each prompt is designed to take an input and produce a predictable output.
  - **Simple Chaining:** Often used when the task can be decomposed into clearly defined, sequential stages.
    > “…a workflow prompt looks like, you have one prompt… take the output, feed it into prompt B… and then you’re done.” citeturn0file0

### Prompt

- **Definition:**  
  A prompt is the input text provided to an LLM to elicit a response.
- **Types:**
  - **Workflow Prompt:** Typically designed for one specific transformation or step.
  - **Agent Prompt:** More open-ended, often includes instructions for the model to use various tools and to iterate until a solution is reached.
- **Importance:**  
  Crafting the prompt well is crucial because it frames how the model perceives the task and what actions it might take.

### Tools and Tool Descriptions

- **Definition:**  
  Tools refer to external capabilities (e.g., web search, code execution, file editing) that can be integrated with an agent.
- **Design Considerations:**
  - **Documentation:** Just as a human engineer requires clear documentation for a function, the model needs clear, well-documented instructions about available tools.
  - **Empathy with the Model:** Developers must “think like the model” to ensure that tool descriptions and prompt instructions are unambiguous and detailed.
    > “…you have to be prompt engineering in the descriptions of your tools themselves… and it’s all getting fed into the same prompt in the context window.” citeturn1file0

## 3. Detailed Discussion Points

### A. Agent vs. Workflow Distinction

- **Autonomy vs. Fixed Steps:**  
  The discussion clarifies that while workflows are straightforward pipelines with predetermined steps, agents are designed to be flexible and autonomous. This means that agents can decide in real time how many iterations or which additional actions are necessary.
- **Real-World Implications:**  
   For example, while a workflow might categorize a user query and then pass it through a fixed series of prompts, an agent may decide to conduct web searches, interact with code, or seek additional clarification dynamically.
  > “…in contrast, an agent prompt will be sort of much more open-ended… giving the model tools or run code and keep doing this until you have the answer.” citeturn0file0

### B. Developer Anecdotes and Lessons Learned

- **Agent Trajectory Insights:**  
   Barry shares an anecdote about monitoring agent trajectories during a computer benchmark (OSWorld). Initially, the decisions made by the model were counterintuitive. By “acting like Claude” (i.e., putting themselves in the model’s shoes), the team was able to better understand and design the agent.
  > “So we decided we're gonna act like Claude and… close our eyes, then blink at the screen, and think, ‘what would I do?’ and suddenly it made a lot more sense.” citeturn1file0
- **Empathy with the Model:**  
  A major takeaway is that successful agent design requires empathy: developers should consider the model’s perspective, including its context and the information it might not possess. This insight should be reflected in both the prompt and the tool descriptions.
- **Prompting for Tool Use:**  
   Erik and Barry note that many developers invest heavily in detailed prompts but neglect the corresponding tool documentation. Since the entire prompt (including tool descriptions) is fed into the model’s context, poor documentation can hinder performance.
  > “…people will put a lot of effort into creating these really beautiful, detailed prompts and then the tools… are sort of these incredibly bare bones… no documentation… how can you expect Claude to use this as well?” citeturn1file0

### C. Hype Versus Reality in Agent Applications

- **Overhyped Aspects:**  
   There is a sentiment that consumer-facing agents are receiving more hype than warranted. For instance, while the idea of an agent autonomously booking a vacation sounds appealing, it may prove to be almost as complex as doing it manually.
  > “Trying to have an agent fully book a vacation for you, almost just as hard as just going and booking it yourself.” 
- **Underhyped Advantages:**  
   On the flip side, agents that save even a minute of time per task can have massive cumulative benefits when scaled across hundreds of tasks. Small efficiencies add up significantly in practice.
  > “I feel like underhyped is things that save people time, even if it's a very small amount of time… that changes the dynamics of now you can do that thing a hundred times more than you previously would.” 
- **Appropriate Use Cases:**  
   Erik also discusses the “sweet spot” where agents are most effective—tasks that are valuable and complex but have a low cost of error or monitoring. This calibration helps developers decide when to implement a full agent rather than a simpler workflow.
  > “I think there's this intersection that's a sweet spot for using agent… tasks that are valuable and complex, but also maybe the cost of monitoring error is relatively low.” 

### D. Motivations for Clear Definitions and Diagrams

- **Customer Clarity:**  
  One motivation for creating detailed documentation and diagrams (as mentioned by Barry and Erik) is the confusion in the industry. Different teams might use various terms for similar concepts. By providing clear definitions and visual aids, they aim to standardize the conversation about agents.
- **Internal Alignment:**  
   Having a unified set of definitions helps when engaging with customers, ensuring that everyone—developers and business users alike—shares a common understanding of the technology.
  > “We walk into customer meetings and everything is referred to as a different term even though they share the same shape… it would be really useful if we can just have a set of definitions and a set of diagrams and code to explain these things.” citeturn1file0


## 4. Practical Guidelines for Developers

### When to Use a Workflow

- **Fixed Tasks:**  
  Use workflows when the task has a well-defined, linear process.
- **Predictability:**  
  When the sequence of steps is known and each step’s output is deterministic.
- **Examples:**
  - Simple categorization or transformation tasks.
  - Predefined pipelines where intermediate validation is required.

### When to Use an Agent

- **Complex, Dynamic Tasks:**  
  Use agents for tasks where the number of steps is uncertain or the process may need to change dynamically.
- **Tool Integration:**  
  When the agent needs to interact with external tools (e.g., web search, code execution) in an iterative manner.
- **Examples:**
  - Customer support applications that require back-and-forth interactions.
  - Situations where minor time savings across repetitive tasks can yield significant efficiency gains.

### Tips for Effective Agent Design

- **Think Like the Model:**  
  Empathize with the model’s perspective by imagining how it perceives the task and what context it might be missing.
- **Detailed Tool Documentation:**  
  Provide clear, human-readable descriptions for each tool. Think of it as writing documentation for a function that another engineer must use.
- **Iterative Prompt Engineering:**  
  Recognize that the overall prompt (including tool descriptions) is critical. A well-constructed prompt leads to better performance and more predictable agent behavior.
- **Monitor Agent Trajectories:**  
   Keep an eye on how the agent behaves in production. Unexpected or counterintuitive actions can be a signal to revisit prompt design or tool instructions.
  > “I think a lot of agent design comes down to… being empathetic to the model and making a lot of that clear in the prompt, in the tool description, and in the environment.”


