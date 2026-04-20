## **TreeWriter: AI-Assisted Hierarchical Planning and Writing for Long-Form Documents** 

**Zijian Zhang**[1] _[,]_[2] , **Fangshi Du**[1] , **Xingjian Liu**[1] , **Pan Chen**[1] _[,]_[2] , **Oliver Huang**[1] , **Runlong Ye**[1] , **Michael Liut**[4] , **Alán Aspuru-Guzik**[1] _[,]_[2] _[,]_[3] _[,]_[5] _[,]_[6] _[,]_[7] _[,]_[8] _[,]_[9] _[,][∗]_ 

> 1Department of Computer Science, University of Toronto, Sandford Fleming Building, 10 King’s College Road, ON M5S 3G4, Toronto, Canada 

> 2Vector Institute for Artificial Intelligence, 661 University Ave. Suite 710, ON M5G 1M1, Toronto, Canada 

> 3Department of Chemistry, University of Toronto, Lash Miller Chemical Laboratories, 80 St. George Street, ON M5S 3H6, Toronto, Canada 

> 4Department of Mathematical and Computational Sciences, University of Toronto Mississauga, 3359 Mississauga Road, Deerfield Hall, ON L5L 1C6, Mississauga, Canada 

> 5Department of Materials Science & Engineering, University of Toronto, 184 College St., M5S 3E4, Toronto, Canada 

> 6Department of Chemical Engineering & Applied Chemistry, University of Toronto, 200 College St. ON M5S 3E5, Toronto, Canada 

> 7Acceleration Consortium, 700 University Ave., M7A 2S4, Toronto, Canada 

> 8Canadian Institute for Advanced Research (CIFAR), 661 University Ave., M5G 1M1, Toronto, Canada 

> 9NVIDIA, 431 King St W #6th, M5V 1K4, Toronto, Canada 

Long documents pose many challenges to current intelligent writing systems. These include maintaining consistency across sections, sustaining efficient planning and writing as documents become more complex, and effectively providing and integrating AI assistance to the user. Existing AI co-writing tools offer either inline suggestions or limited structured planning, but rarely support the entire writing process that begins with high-level ideas and ends with polished prose, in which many layers of planning and outlining are needed. Here, we introduce TreeWriter, a hierarchical writing system that represents documents as trees and integrates contextual AI support. TreeWriter allows authors to create, save, and refine document outlines at multiple levels, facilitating drafting, understanding, and iterative editing of long documents. A built-in AI agent can dynamically load relevant content, navigate the document hierarchy, and provide context-aware editing suggestions. A within-subject study ( _N_ = 12) comparing TreeWriter with Google Docs + Gemini on long-document editing and creative writing tasks shows that TreeWriter improves idea exploration/development, AI helpfulness, and perceived authorial control. A two-month field deployment ( _N_ = 8) further demonstrated that hierarchical organization supports collaborative writing. Our findings highlight the potential of hierarchical, tree-structured editors with integrated AI support and provide design guidelines for future AI-assisted writing tools that balance automation with user agency. 

**Correspondence:** Alán Aspuru-Guzik at alan@aspuru.com **Code:** https://github.com/aspuru-guzik-group/TreeWriter 

**==> picture [43 x 32] intentionally omitted <==**

**==> picture [44 x 24] intentionally omitted <==**

**==> picture [44 x 11] intentionally omitted <==**

**==> picture [19 x 17] intentionally omitted <==**

**==> picture [43 x 10] intentionally omitted <==**

## **1 Introduction** 

AI-assisted long-form writing presents unique challenges that extend beyond simple text generation, as this task requires managing information at scales that exceed human working memory [27]. Unlike short-form writing, where authors can maintain a coherent mental representation of the entire text, long documents such as research grants, reports, or books demand sustained attention to structure, coherence, and dependencies across multiple sections. Writers often employ external cognitive scaffolds to mitigate these limitations. Common strategies include repeated readings to construct a cognitive map of the global text structure, 

1 

**==> picture [472 x 229] intentionally omitted <==**

**----- Start of picture text -----**<br>
Related work Background<br>• ______________________ _________________________<br>• ______________________ _________________________<br>Background Export _________________________<br>• ______________________ ________________________________________________________________________ __________________<br>•• ____________________________________________ __________________________________ + & Leaf nodesExports  Related work_________________________<br>Export _________________________<br>____________________________________<br>____________________________________ _________________________<br>__________________________________ Recent progress __________________<br>_________________________ Recent progress<br>_________________________<br>_________________________<br>__________________<br>_________________________<br>+ __________________<br>1 Tree View 2 Linear View<br>**----- End of picture text -----**<br>


**Figure 1** TreeWriter enables users to view and edit their documents in two complementary views: (1) the **tree view** and (2) the **linear view** . In the tree view, users develop outlines at each node, expand them into text for the final document, or split them into child nodes for further elaboration. Users can use integrated AI features to maintain coherence and consistency across related nodes. The hierarchical structure formed in this process supports easy navigation and multi-level editing. The linear view compiles complete sections by traversing the subtrees and concatenating the exported content from nodes sequentially. A chat-based writing assistant with a scoped context is available in both views, offering node-level writing and revision suggestions. 

adding annotations to the actual texts, and using professional writing tools. In collaborative contexts, the difficulty shifts toward sustaining a shared cognitive representation of the document, which requires aligning understanding across contributors. Collectively, these constraints underscore the need for AI systems that extend beyond local text generation, facilitating the higher-order processes of memory management, which is essential for effective long-form writing. 

The recent advances in large language models (LLMs) create an opportunity to solve these challenges [42, 43, 23, 57, 3, 10]. Text can now be rapidly and automatically reconfigured across multiple representational forms, such as bullet points, continuous prose, or images, to facilitate human understanding and operating [61]. Current AI-assisted writing tools [47] can be broadly categorized into two paradigms: _inline suggestion editors_ that provide immediate text modifications within linear documents [60, 7, 46, 59], and _conceptual editors_ that support planning and organization beyond direct text manipulation [54, 62, 51]. While inline editors offer immediate assistance, they rely heavily on the ability of LLMs to reconstruct the hidden conceptual structure behind texts. Existing conceptual editors, although they expose the conceptual structure to the user, still lack methods to build further hierarchies of abstraction that support the efficient editing of longer documents (See Section 2). 

To address these limitations, we introduce **TreeWriter** , a hierarchical writing system that integrates AI assistance with tree-based document organization. TreeWriter represents a document as a hierarchical tree, allowing authors to iteratively construct higher-level abstractions from detailed content with the help of AI, forming a multi-level representation of their document. Based on the tree structure, TreeWriter integrates a writing assistant that can operate on these levels, offering suggestions at multiple levels that maintain consistency between high-level plans and low-level text, thereby supporting both high-level conceptual reasoning and actual document editing throughout the writing process. 

The development of the system was informed by a formative study involving six researchers who regularly produce long-form content. The study revealed three key challenges in AI-assisted writing: (1) structuring ideas and transforming them into text, (2) maintaining consistency across sections, and (3) verifying the 

2 

accuracy and trustworthiness of AI-generated content. These findings informed the design of TreeWriter, focusing on three key objectives: supporting idea structuring and iterative drafting, ensuring cross-section consistency, and facilitating trustworthy AI collaboration. 

We evaluated TreeWriter through both a comparative lab study and a two-month field deployment. In the lab study, we compared TreeWriter with Google Docs [20] equipped with the Gemini assistant across two tasks: modifying a long article (approximately 4,000 words) and writing a new article (approximately 800 words). The results demonstrate that TreeWriter effectively supports both editing and drafting of long-form documents while allowing users to retain control over AI-generated content. In the field deployment, participants utilized TreeWriter in real-world collaborative writing projects over two months. Findings from this study suggest that TreeWriter can enhance team collaboration in co-authoring long documents. 

The main contributions of this work are: 

1. The design and implementation of **TreeWriter** , a novel writing system that demonstrates how a hierarchical document structure can be integrated with AI assistance to overcome the limitations of existing writing tools. 

2. A **mixed-methods evaluation** , which includes a controlled in-lab user study and a field deployment, demonstrates that TreeWriter is more effective than a standard linear editor for long document editing, creative writing and collaborative writing in many dimensions. 

3. A set of **design recommendations** for developing AI-assisted writing tools that support long-form document editing and balance automation with user control, informed by insights from our formative study and evaluation findings. 

## **2 Related Works** 

We organize related prior work along several dimensions: (1) the cognitive process of writing, (2) AI-assisted writing interfaces, and (3) document structure and hierarchical organization tools. By synthesizing insights from these areas, we identify recurring gaps between what cognitive theories of writing suggest and what existing tools provide—namely, the lack of support for hierarchical planning, cognitive offloading, and intelligent text manipulation at scale. These gaps directly motivate the design of TreeWriter. 

## **2.1 The Cognitive Process of Writing** 

Hayes and Flower’s foundational cognitive model conceptualizes writing as a complex cognitive activity characterized by interwoven cycles of planning, text generation, and revision [17]. Empirical studies have shown that in the planning stage, writers form hierarchical goals and often revisit these goals throughout the writing process [16]. In the text generation stage, writers translate ideas from the planning stage into concrete textual content [27]. In the revision stage, writers engage in surface-level editing, such as spelling and paraphrasing, as well as more complex structural changes that revise the text’s meaning [14, 53]. During these cognitive processes, semantic, syntactic, and lexical information is stored on the fly within the writer’s working memory [41], so writers with greater working memory capacity tend to produce more fluent and coherent texts [27]. 

We draw two conclusions from this cognitive model of writing. First, because writing is a non-linear process, writers often need to revisit and revise intermediate content, such as sketches of arguments, structures, goals, and sub-goals, produced during the earlier stages of the writing process. These cognitive scaffolds guide the writing process, but do not appear in the final text. Therefore, preserving these cognitive scaffolds throughout the writing process is essential, as it enables writers to revisit and reflect on them. Second, because working memory is a bottleneck, interfaces that reduce the load on working memory have a positive impact on how writers plan and revise. Consequently, an effective writing tool must both support the capture and persistence of cognitive scaffolds and provide mechanisms that reduce cognitive load [29]. 

3 

## **2.2 AI-assisted Writing Systems** 

A growing body of work has introduced systems that support human–AI co-writing [47]. These systems span the stages of the writing process — including planning [51, 62, 54], text generation [54, 59, 60, 7, 38, 62], and revision [46, 38, 59, 60] — and are applied across diverse writing contexts. To facilitate analysis, we classify these works into two categories: (1) those that operate directly within the text to suggest edits or continuations, and (2) those that assist in planning, structure and idea guidance beyond surface-level edits. We refer to the first category as _Inline Editors_ and the second as _Conceptual Editors_ . 

## **2.2.1 Inline Editors** 

Inline Editors embed LLM-based features directly within the document editing interface, providing users with immediate suggestions and modifications to their selected text content. Earlier works, such as Wordcraft [60] and CoPoet [7], enable users to invoke LLMs to generate continuations based on selected text and/or a prompt. Wordcraft also allows for rewriting or elaborating in place within the document. Recent works have provided more sophisticated methods for LLM-assisted interactions within text. TextoShop [38] offers features such as rewriting, extending, shortening, style change, grammar checking, and text merging through a drawing-software-inspired interface. GhostWriter [59] also provides rewriting and extending features, but enhances these features through personalized style and context settings. ABScribe [46] focuses on the revision stage of writing by providing an interface for generating, storing, and comparing multiple alternative versions of a selected passage adjacent to the original text. This allows the user to explore different phrasings and select the most appropriate option for their context. 

Inline approaches face limitations. While these tools provide immediate suggestions for improvement and enable authors to refine or accept AI-generated rewrites on the spot, they typically operate at the sentence or paragraph level without considering broader document structure or hierarchical relationships beyond the user-selected content. Therefore, the common interaction pattern in current inline editors—manually selecting relevant text, inserting prompts, and regenerating content—does not generalize to long-form writing, as it requires repeated modifications that become tedious as the document size grows. Although users can technically select the entire document for LLM modification, they typically prefer fine-grained control over the generated content instead of handing it all to the model [26, 47]. 

## **2.2.2 Conceptual Editors.** 

In contrast to inline editors, conceptual editors support writing through higher-level structures and interactions in addition to inline text edits, focusing on planning, organization, and creative exploration. To facilitate idea exploration, Luminate [54] utilizes a user-provided writing prompt to generate a collection of drafts, each comprising a particular set of labels belonging to dimensions such as setting or tone. This provides a structured view for exploring alternative drafts, as the user can organize drafts into clusters that share the same labels. Script&Shift [51] offers a layered editing interface to support idea exploration. In Script&Shift, the user can create drafts that are stored as draggable editor windows called layers within a shared workspace. The system enables quick iterations on drafts by providing layer-level operations such as prompt-based rewriting, inter-layer comparison, child layers, and stacking layers together to visualize the final linear document. VISAR [62] supports visual writing planning by providing an interface that displays high-level arguments, discussion points, supporting evidence, and counterarguments as nodes in a tree connected by logical relationships. This allows the user to edit the draft by adding or modifying nodes in the tree. 

While these systems elevate writing beyond direct text manipulation, they often operate within limited levels of abstraction. For instance, VISAR’s abstraction model is tightly coupled to argumentative writing and lacks generality for broader forms of creative or expository writing. Script&Shift allows adding child layers iteratively; however, its linked layers lack explicit abstraction relations, functioning more as annotations than abstractions, which limits the form of hierarchical structure. As a result, existing conceptual editors still fall short of supporting representing and operating multi-level abstraction of documents. 

4 

||Wordcraft|Luminate|Textoshop|Script&Shift|VISAR|**TreeWriter**|
|---|---|---|---|---|---|---|
||[60]|[54]|[38]|[51]|[62]|**(This work)**|
|Document structure|Linear|Linear|Linear|Node+Linear|Node+Linear|**Node+Linear**|
|AI suggestion|Inline|Inline|Inline|Inline|Inline & Nodes|**Inline & Nodes**|
|Node relation|–|–|–|Arbitrary|Logic-based|**Abstract-based**|
|Progressive disclosure|–|–|–|Partial|Partial|**Yes**|



**Table 1** Comparison of AI writing tools. TreeWriter uniquely employs abstract-based node relations, where we prompt the user and agents to make the parent nodes contain the outlines of their child nodes. This enables progressive disclosure for both human and AI agents and therefore supports scalable interaction with long documents. 

## **2.3 Document Structure and Organization Tools** 

Empirical studies have shown that writers who adopt a structured approach to writing consistently produce texts with better overall quality [52, 11, 35, 30]. For instance, students who repeatedly used electronic outlining tools developed clearer text structures and reported lower mental effort during writing [11]. Similarly, students who plan with a structured outline that embeds the main arguments performed better than those who use an unstructured list of ideas [35]. When writing tasks are presented as a set of sub-goals, it is also found that fine-grained goals reduced cognitive load and improved reasoning compared to clustered goals [30]. 

Consistent with these findings, several existing tools incorporate various features that support a structured approach to writing. Scrivener [36] provides corkboard and outlining features. L[A] TEX-based systems such as Overleaf [24] supports hierarchical section management. However, these tools either incorporate no AI functionality or restrict their assistance to in-line text improvements, lacking mechanisms that leverage inter-section relationships to provide structure-aware AI assistance. 

TreeWriter contributes to this landscape by combining the strengths of both inline and conceptual editing paradigms with a tree-based hierarchical interface. In the document trees of TreeWriter, non-leaf nodes preserve cognitive scaffolds such as goals, outlines, and summaries, while the exported content, mostly from leaf nodes, corresponds to the final textual output. This design enables structured planning and reduces cognitive load when switching contexts between planning, text generation, and revision. Unlike existing inline editors that operate on individual text segments, TreeWriter features a chat-based, agentic writing assistant that operates on the nodes of the tree. This design enables TreeWriter to automate the “select–prompt–generate” interaction pattern across the document by leveraging its agentic capabilities to perform multi-step interactions to load the essential context and generate document content at both high and low levels. 

## **3 Formative Study** 

To better understand the challenges of long-form writing and attitudes toward AI-assisted writing, we conducted a formative study with researchers who regularly produce long documents. Insights from this study directly informed the design of TreeWriter by highlighting pain points in document writing and surfacing user preferences for AI-assisted workflows. 

## **3.1 Study Design and Participants** 

We recruited six participants through academic networks and personal connections (their anonymized information is available in Appendix A). Participants included graduate students and postdoctoral researchers who routinely produce documents exceeding ten pages in length, such as research articles, grant proposals, and technical reports. The study was conducted through an online structured survey that combined open-ended interview questions with Likert-scale questions. The complete list of questions is available in Appendix B. 

The interview questions focused on two main areas. The first explored the participants’ background and existing practices in academic writing. The participants were asked to describe their typical processes for working on long documents, the most significant challenges they face, and what features they wish for in writing tools to better support this work. The second area covered their perceptions of AI in writing. This section 

5 

explored participants’ prior use of AI tools for writing tasks, the impact those tools had on their workflow, and their perspectives on what constitutes responsible AI support for collaborative writing. Likert-scale questions were aligned with these topics and served as a complementary supplement to the interview questions. 

## **3.2 Findings** 

Analysis of responses revealed three recurring challenges that shaped the design of TreeWriter: 

## **3.2.1 C1. Structuring Ideas and Transforming Them into Text** 

Participants described writing as an iterative cycle of outlining, expanding, and refining. F1 notes that “I usually start with a one-page draft and ask the AI to expand it into ten pages—it saves about a week of work.” A typical process (n=4) involved sketching a high-level structure (e.g., a short draft, outline, or mind map), drafting sections in collaboration with AI tools, and then engaging in human-led refinement to improve style, accuracy, and formatting. This iterative loop enabled authors to focus on high-level ideas while offloading more mechanical drafting tasks; however, it also highlighted gaps in tool support for structuring and navigating evolving ideas. Most participants (n=5) agreed or strongly agreed that writing long documents is often frustrating and that they frequently lose track of the overall structure, highlighting the need for writing tools that support high-level organization. For instance, one participant (F4) wished for, “I would like a more interactive writing tool that can guide me through the document-building process — taking a high-level outline as input and prompting me to fill in details.” 

## **3.2.2 C2. Maintaining Consistency Across Sections** 

Ensuring coherence in terminology, argumentation, and narrative flow across sections was a persistent challenge. As one participant (F2) explained, “It’s challenging to correlate the context and keep high-quality writing consistent from beginning to end.” Participants (n=2) mentioned that maintaining coherence is challenging in long-form documents. Participants also envisioned tools to better preserve coherence across sections, for example, “a helper that can maintain long context and summarize what’s been written” (F2). 

## **3.2.3 C3. Verifying Accuracy and Trustworthiness** 

Participants expressed concerns over the factual accuracy of using AI-generated text. As one participant warned, “AI tools sometimes cite fake or incorrect references, which causes a lot of trouble” (F3). They reported instances of hallucinated citations, misleading references, or technically inaccurate claims, all of which demanded substantial verification effort. Likert responses indicate participants were generally comfortable using AI to assist writing but showed more caution about trusting AI to draft content autonomously, highlighting the importance of verifiable outputs. As another participant emphasized, “Responsible AI should at least verify the authenticity and reliability of the references it provides” (F6). 

## **3.3 Design Goals** 

Guided by insights from our formative study (Section 3), we designed TreeWriter to address the central challenges of long-form writing while ensuring responsible AI support. The resulting design goals map directly to the three challenges identified in the study. 

- **DG1** : **Support Idea Structuring and Iterative Drafting.** Enable writers to organize their ideas hierarchically and develop them progressively into text. This involves creating outlines, structuring content across multiple levels, and supporting iterative expansion from high-level concepts to detailed paragraphs. 

- **DG2** : **Ensure Consistency Across Sections.** Promote coherence throughout a long document by utilizing the hierarchy. This includes detecting inconsistencies at each level and providing guidance that maintains continuity across sections. 

- **DG3** : **Enable Trustworthy and Transparent AI Collaboration.** Ensure that AI support is accurate, reliable, and controllable. This involves presenting AI suggestions transparently, allowing authors to review and revise outputs, and maintaining accountability for changes made with AI assistance. 

6 

**==> picture [472 x 269] intentionally omitted <==**

**----- Start of picture text -----**<br>
2 3<br>1<br>**----- End of picture text -----**<br>


**Figure 2** TreeWriter’s interface consists of three columns: (1) a **tree navigator** on the left for organizing the document structure, with a view switcher at the top for changing the middle-column view; (2) the **middle column** , which supports two complementary modes: a **tree view** (currently shown) for hierarchical editing by displaying the children of a parent node, and a **linear view** for previewing the composed text of that node’s subtree. A floating toolbar at the top allows navigation to higher-level nodes and searching within the document; (3) a chat-based **writing assistant** and AI editing buttons on the right for AI-assisted document editing. 

These goals underpin both the hierarchical editor and its AI features, ensuring that TreeWriter provides helpful and trustworthy support for long-form writing. 

## **4 TreeWriter** 

Based on the design goals, we designed and implemented **TreeWriter** . TreeWriter’s interface is organized into three columns (See Figure 2): the left column is a tree navigator where nodes can be reordered through drag-and-drop, the middle column displays document content, and the right column hosts the chat-based writing assistant and AI-powered editing buttons. The middle column supports two complementary views: a _tree view_ for structural and conceptual editing, and a _linear view_ for previewing the final document. These two views can be switched by the buttons at the top of the left column. 

## **4.1 Tree View** 

In the tree view, the middle column displays the children of a specific parent node. Each node is shown as a card containing a title and editable content. Navigation controls at the bottom of each card enable the user to navigate to the parent, navigate to the node’s children, add a new child, or delete the current node. Nodes can also be reordered via drag-and-drop. When the user clicks a node, a blue dot will appear at the top-left corner of the node to show it is “selected”, which provides context to AI assistance. 

This design supports a multi-level hierarchical representation of documents: parent nodes summarize their children, and children elaborate on their parent. Such hierarchical decomposition allows users to iteratively refine high-level ideas into detailed content while maintaining the conceptual structure behind the document. 

In order to retain the outlines, TreeWriter does not directly output all the content in the nodes to the final 

7 

**==> picture [472 x 182] intentionally omitted <==**

**----- Start of picture text -----**<br>
This is too long Outline is easy to read I maintain consistency<br>AI in Health Care AI in Health Care Modification on  AI in Health Care<br>_________________________ • ______________________ high-level outlines • ______________________<br>_________________________ • ______________________ • _________________<br>_________________________ • ______________________ • ______________________<br>_________________________ • ______________________ • _______________<br>3<br>_________________________<br>____________ Summarize /<br>_________________________ 2 Update 4<br>_________________________ Update<br>_________________________<br>_________________________ AI for Drug Discovery Clinical Diagnosis  Ethical Concerns<br>_________________________<br>Decompose ______________________ ______________________ ______________________<br>________________<br>______________________ ______________________ ______________________<br>_________________________<br>__________________ ______________________ ______________________<br>_________________________<br>_______________________ 1 _____________________ ______________________<br>**----- End of picture text -----**<br>


**Figure 3** AI-assisted abstraction creation. (1) TreeWriter lets the user freely write within a node, which can grow to any length. Later, the user can split it into reasonable chunks using the **“Split into subsections”** button, which leverages an LLM to generate child nodes that collectively cover the original text. (2) Once the text has been split, the user can use the **“Generate outline from children”** button to rewrite the parent node as a concise outline of its children. This “split&summarize” process reduces large nodes and makes the whole tree more reader-friendly. This function can also be used after substantial edits to the children, so the parent content remains in sync with the children. (3) Users can revise a section at a high level by modifying its root node and then use AI to propagate these changes to the subtree. (4) In response to a chat request, TreeWriter’s writing assistant can update the child nodes to ensure that the final content reflects the revised outline. This makes it convenient to maintain consistency between the higher-level outline and the lower-level realization during the revision stage of writing. The writing assistant can also be asked to maintain the coherence of the child nodes. 

document. Node content connects to the final document in two ways: (1) authors can add an _export block_ at the end of a node’s content (See Figure 5), which will be included in the final document, or (2) if no export block is specified, leaf-node content is exported by default. This flexible mechanism helps users distinguish between structural notes and publishable text, ensuring clarity for collaborators and AI assistance without cluttering the final output. 

## **4.2 Linear View** 

The linear view displays a section of the final document by concatenating exported content from all nodes (See Figure 4). Users can scroll through this view to inspect how the final text will appear, and use a toggle button at the top of the right column to switch between tree and linear views. 

Within the linear view, each node supports a floating menu which appears when the user clicks. The menu has three buttons: (1) navigate to the corresponding node in the tree view, (2) display only the subtree of the clicked node for focused reading, and (3) toggle edit mode to show and allow the user to directly edit the whole node content, including the parts not exported. 

## **4.3 Complementary Roles of the Two Views** 

In TreeWriter, the tree view serves as the primary interface for writing and organization, while the linear view supports holistic inspection of the evolving manuscript. By combining the tree view and the linear view, TreeWriter reduces cognitive load, preserves structural clarity while maintaining compatibility with traditional formats of documents. 

## **4.4 AI-Powered Editing Buttons** 

It can be laborious for users to manually maintain the consistency of the hierarchical outlines in the tree. To assist users in building and maintaining the abstraction hierarchy (DG1, DG2), TreeWriter integrates several 

8 

**==> picture [472 x 347] intentionally omitted <==**

**----- Start of picture text -----**<br>
Editor view<br>Linear view<br>3<br>2<br>1<br>1<br>**----- End of picture text -----**<br>


**Figure 4** Transform from Tree view to Linear view. TreeWriter connects the node-based editor to the final document through the export blocks. Only the content in the export block and the content in a leaf node without an export block appear in the final document. In the linear view, the tree (or a selected subtree) is linearized by a preorder traversal and the content exported from each node is listed in the view. (1) Only the exported content of the nodes is included in the final document. (2) In the linear view, a menu appears when the user clicks on each exported content, which allows the user to focus on the subsection from that node, jump to the corresponding node in the tree view or directly edit that node’s content in place. (3) A navigator can be used to expand the scope of the linear view to the ancestors of the current section. 

9 

**==> picture [472 x 325] intentionally omitted <==**

**----- Start of picture text -----**<br>
Add  Check  Improve  Make it<br>detail grammar coherence shorter<br>4<br>1 Generate Paragraph<br>2 3 5<br>Add connecting  Check export  Check<br>sentence to the  matches the  consistency to<br>4 previous section outline the parent node<br>**----- End of picture text -----**<br>


**Figure 5** AI-assisted paragraph generation in TreeWriter. Users are encouraged to create and save both the outline and the corresponding prose in the nodes. (1) The user can draft an outline first, then click **“Generate paragraph”** to create an export block with a draft paragraph ready for inclusion. (2–3) Each export block provides two sync buttons: generating a paragraph from an outline or generating an outline from a paragraph. These features allow flexible editing from either direction while maintaining consistency between the outline and the paragraph. (4) A chat-based writing assistant can refine both outlines and paragraphs by the user’s instruction, with awareness of the context of the node in the document. (5) When the writing assistant modifies existing content, a confirmation dialogue with a difference viewer will let the user review, accept, or adjust the proposed changes. 

10 

AI-powered editing buttons, which are displayed at the bottom of the right column. The editing buttons include: 

- _Split into subsections_ : Breaks down the content of the selected node into several child nodes of it to improve readability and facilitate further elaboration on them. 

- _Generate outline from children_ : Replace the content of the selected node with a summary of the child nodes of it to create abstraction or maintain parent-child consistency. 

- _Generate paragraph_ : Adds an export block to the end of the selected node with a paragraph generated from the outlines in the block. 

Additionally, within the export block, there are two buttons (See Figure 5 (2-3)): 

- _Generate paragraph_ : Generates a paragraph based on the outlines in the node. 

- _Generate outline_ : Generates an outline based on the exported paragraph in the node. 

When an editing button is clicked, the content and context of the node are sent to an LLM to generate the corresponding output. The LLM is prompted to preserve the existing content whenever possible (See Section I.2 for detailed prompts). After the content is generated, a confirmation dialogue will appear, allowing users to review and edit the generated content before accepting it, reinforcing DG3. 

## **4.5 Agentic writing assistant on Tree** 

In the right column of the interface, TreeWriter provides a chat interface for users to interact with the LLM-based writing assistant (See Figure 6). The chat interface allows users to ask questions, seek suggestions, and discuss ideas with the assistant. The chat history is preserved unless the user clicks the reset button, enabling context-aware conversations that build upon previous interactions. 

The writing assistant is constructed by assigning proper context and tools to an LLM-based agent. The list of tools that are available to the agent can be found in Table 2, which includes making editing suggestions to the user, loading node content and searching nodes by keywords. The context of the agent includes the title and content of the selected node and its parent. The titles and ID of the sibling and child nodes of the selected node are also presented in the prompt of the agent to enable a dynamic loading of their content based on their ID. 

All this contextual information enables the writing assistant to generate editing suggestions that are not only locally fluent but also globally consistent with the surrounding nodes. The user can request **localized edits** targeting a single node—for example, asking the assistant to check whether a node’s content aligns with that of its parent, siblings, or children (DG2). The tree structure also supports efficient high-level modifications: after revising a high-level node, the user can instruct the assistant to update its child nodes accordingly, achieving **high-level conceptual editing** (DG1). Moreover, the user can direct the assistant to focus on a specific concept within the document, prompting it to locate related nodes through keyword search or structural traversal (e.g. beam search) and then make modification suggestions (DG1). 

|**Tool name**|**Description**|
|---|---|
|_Load Node Content_|Loads a node’s content into the context by its ID.|
|_Load Node Children_|Loads the ID and title of the child nodes of a node into the context.|
|_Suggest New Title_|Suggests a new title of a node for the user to review.|
|_Suggest New Content_|Suggests a new version of the content of a node for the user to review.|
|_Suggest New Child_|Suggests a new child to a certain node for the user to review.|
|_Search by Keyword_|Searches for nodes that contain a given keyword from the whole tree.|



**Table 2** Available tools in TreeWriter’s writing assistant. These tools enable the assistant to modify or add content to the nodes. 

11 

**==> picture [472 x 423] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 Modify content 2 Modify title<br>3 Search & Modify 4 Add children<br>**----- End of picture text -----**<br>


**Figure 6** Writing assistant. TreeWriter includes an agentic writing assistant that is aware of the context of the nodes to edit. The assistant can use tools to dynamically load the contents of the node and assist the user in editing the tree. (1,2) The assistant can provide editing suggestions on the content and title of nodes based on the user’s requests. Suggestions are displayed as blocks with a blue background in the chat interface; when clicked, a confirmation dialogue appears to help the user confirm the change. (3) The assistant can search for nodes by keywords, enabling it to locate and edit nodes based on the user’s request. The search results are displayed in the chat interface, and users can navigate directly to the nodes by clicking. (4) The assistant can also suggest adding new child nodes; the user can accept these suggestions by clicking blocks with a green background. 

12 

## **4.6 Confirmation and Versioning** 

To uphold transparency and trust (DG3), all AI-generated editing suggestions undergo review through an interactive confirmation dialogue before being applied (See Figure 5). The dialogue employs a two-column layout that displays the original text on the left and a detailed difference view highlighting proposed AI changes on the right, enabling authors to quickly understand what is going to be modified. 

An optional _edit mode_ empowers authors to modify AI-generated editing suggestions before acceptance; the difference view updates dynamically once edit mode is deactivated, showing the final changes relative to the original content. All accepted revisions are systematically archived with descriptive labels in a comprehensive version history, enabling easy restoration, side-by-side comparison, and safe experimentation without risk of data loss. 

## **4.7 Implementation detail** 

We used TypeScript, React and Material UI[1] to develop the user interface. We used the TipTap[2] framework for building the editor. To enable collaborative editing in TreeWriter, we integrated Yjs [40], allowing multiple users to edit simultaneously. On the backend, we employed TypeScript and Express.js. The writing assistant is implemented by the Vercel AI SDK[3] and uses GPT-5 [43] from OpenAI as the backbone model. The AI-powered editing buttons utilize GPT-4.1 [42] for quicker response to users. 

## **5 Comparative Lab Study** 

We conducted a user study to evaluate the effectiveness of TreeWriter in supporting long-form AI co-writing compared to a baseline system. The study aims to answer the following research questions: 

- **RQ1** : (Edit existing long document) How does TreeWriter, with its AI-assisted structure and writing assistance, help users edit and reorganize existing long-form documents? In particular, does the AI assistance help users work more efficiently and maintain a clearer sense of structure and consistency across sections compared to baseline tools? 

- **RQ2** : (Draft from ideas to document) How does TreeWriter support users in transitioning from brainstorming and outlining to drafting coherent prose? In particular, does it help users explore and organize ideas more effectively and perceive greater support from AI assistance compared to baseline tools? 

- **RQ3** : (Authorial control, transparency, and trust) Does TreeWriter increase authors’ perceived and behavioural control over AI-generated content, improve transparency of changes, and reduce verification effort and perceived risk of inaccuracies, resulting in outputs that better reflect the author’s intent? 

## **5.1 Participants and Procedure** 

We recruited 12 participants (P1-P12; 2 women, 10 men) through the mailing lists and Discord channels of student clubs at a research-focused Tier 1 university in North America. Detailed information about the participants can be found in Appendix A. The participants include: five undergraduate students, three master’s students, and four PhD students, with representation from a diverse set of disciplines (computer science, law, and chemistry). All participants reported prior experience with co-writing using AI. Each study session lasted approximately 2.5 hours, and the participants received compensation of $50 CAD. 

## **5.2 Study Design** 

Each study session consisted of two main tasks: **Article Modification** and **Creative Writing** . In the article modification task, the participant is provided with a 4,000-word article on AI’s implications for the world 

> 1https://mui.com/ 

> 2https://tiptap.dev/ 

> 3https://ai-sdk.dev/ 

13 

and is asked to complete six editing sub-tasks on it. In the creative writing task, the participant is given a topic on AI and is asked to write an 800-word essay. We adopted a within-subjects design, in which each participant completed the tasks using both TreeWriter and a control (Google Docs + Gemini). 

The tasks were chosen to fit within the time constraints of each user study session and to hold real-world value. They were modelled after common university-level writing assignments, and the topics were selected to ensure that an average participant would have a moderate amount of familiarity with the subject matter. 

The user study session was conducted as follows (detailed procedure can be found in Appendix D): 1. Introduction to the study, consent, and setup recording. 2. A random tool was set up and introduced through an introductory video. 3. The participants were then asked to complete the article modification task on an article, followed by the creative writing task on a topic. After each task, the participant filled out a post-task questionnaire. This sequence was designed to minimize the cognitive overhead associated with switching between editing and generative modes. 4. The participant then completed the same tasks on the other tool, article, and topic. 

## **5.3 Control** 

We chose Google Docs with Gemini [20], a widely adopted linear document editor that integrates AI functionalities as the baseline system (control group). Gemini offers a chatbot-style interface comparable to TreeWriter, leveraging the document as context to provide writing suggestions. It allows participants to insert AI-generated text directly into the document and to revise selected text through natural language instructions. More details on this can be found here [21, 22]. 

## **5.4 Measures** 

In the post-task questionnaire, participants rated their agreement with seven statements following the article modification task and five statements following the creative writing task. All questions were rated on a 7-point Likert scale (1 = Strongly Disagree, 7 = Strongly Agree). For the article modification task, participants evaluated the helpfulness of AI features, the system’s support for breaking sections into subsections, control over edits, creating summaries from detailed content, expanding outline points into paragraphs, propagating conceptual changes, and obtaining a clear structural overview. For the creative writing task, participants assessed the helpfulness of AI features, support for building and refining document structure, assistance in exploring and developing initial ideas, effectiveness in expanding outline points into paragraphs, and overall productivity. Detailed questions can be found in Appendix C. 

To complement our custom questionnaire, we administered the Creativity Support Index (CSI) [9] in both conditions. The CSI evaluates six dimensions: Exploration, Expressiveness, Immersion, Enjoyment, Results Worth Effort, and Collaboration. Because our tasks were single-author and non-collaborative, we omitted the Collaboration dimension. Participants rated each dimension on a 10-point Likert scale, and we computed per-dimension scores and an overall CSI score following the published procedure. CSI provided a validated, tool-agnostic measure of perceived creativity support, spanning ideation and expression (Exploration, Expressiveness), engagement (Immersion, Enjoyment), and efficiency (Results Worth Effort). 

After completing two tasks, we also administered a NASA-TLX questionnaire [25] to assess cognitive load in both tasks under both conditions. This provides a standardized measure of mental demand, physical demand, temporal demand, performance, effort, and frustration associated with the writing tasks. 

## **5.5 Results** 

## **5.5.1 How TreeWriter helps editing long documents (RQ1)** 

Quantitative results from Task 1 indicate that participants generally found both editors helpful for AI-assisted editing, with TreeWriter achieving consistently higher scores on most questionnaire items (See Figure 7). TreeWriter outperformed Google Docs in ratings for _AI helpfulness_ (6.33 vs. 4.83), _breaking sections_ (6.08 vs. 4.42), _controlling edits_ (5.0 vs. 3.75), _creating summaries_ (6.33 vs. 5.83), _expanding outlines_ (6.17 vs. 5.33), _propagating changes_ (5.5 vs. 4.58), and _providing a structural overview_ (5.92 vs. 5.17). These results suggest 

14 

**==> picture [472 x 155] intentionally omitted <==**

**----- Start of picture text -----**<br>
AI Helpfulness Break Sections Control Edits Create Summary Expand Outline Propagate Changes Structure Overview<br>7 7 7 7 7 7 7<br>6 6 6 6 6 6 6<br>5 5 5 5 5 5 5<br>4 4 4 4 4 4 4<br>3 3 3 3 3 3 3<br>2 2 2 2 2 2 2<br>1 1 1 1 1 1 1<br>GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter<br>User Survey Rating (higher is better)<br>**----- End of picture text -----**<br>


**Figure 7** Participant ratings of the _modification experience_ in the modification task, comparing TreeWriter and Google Docs across seven dimensions: helpfulness of AI features, breaking sections into subsections, control over edits, creating summaries from detailed content, expanding outline points into paragraphs, propagating conceptual changes, obtaining a clear structural overview. Ratings were given on a 7-point Likert scale (higher is better). Results show that TreeWriter consistently achieved higher scores with smaller variation, indicating a more effective and consistent AI-assisted editing experience. 

that TreeWriter offers a more effective AI-assisted editing experience, with higher ratings and less or similar variation across participants. 

Qualitative feedback highlighted two main strengths. First, participants emphasized that TreeWriter’s hierarchical structure improved organization and navigation, particularly for managing large and complex documents. As one participant explained, “The hierarchical structure presents a more intuitive view compared to Google Docs, as a tree structure is more organized than just a linear structure” (P11). The response from another participant (P1) reinforced this advantage: “[TreeWriter is] easy to manage large intricate documents and easy to navigate - no need to keep scrolling.” 

Second, participants appreciated that TreeWriter’s writing assistant could operate across multiple levels of granularity, supporting both local refinements and global restructuring. One participant noted, “The structural nodes allowed me to control the scope and granularity of changes wanted.” Another highlighted the global utility: “The overall AI Assistant is smart enough to look over the tree for all prompts, allowing both granular and coarse editing together.” Together, these features enabled participants to not only polish sentences but also reshape sections and maintain coherence across the document. 

During the study, we also observed a strong tendency for participants to rely on the writing assistant to complete tasks without manually navigating the document. The tendency is observed in both TreeWriter and the control system. A common strategy was to use a search to locate a section, then delegate most of the editing to the writing assistant. In tasks without a clear edit location, participants often prompted the assistant to search the tree itself; for instance, one participant (P9) directly instructed the writing assistant to scan from the root node, which successfully identified the correct location without requiring manual navigation. This demonstrates how our work effectively leverages the tree structure and node-level outlines to target edits accurately. However, TreeWriter also presented drawbacks. While many praised the tree view, some participants found it overly complex for short documents and unfamiliar compared to Google Docs’ linear interface. As one participant put it, “The hierarchical organization is too complex when the total text document is small, and I need to keep clicking to expand” (P1). Another remarked, “TreeWriter’s interface requires extra time to learn and may feel less intuitive at first” (P5). In addition, a few participants (n=2) noted occasional AI hallucinations, which undermined trust in the automation. 

## **5.5.2 How TreeWriter helps draft new documents (RQ2)** 

TreeWriter was frequently described as effective for initiating new writing projects, helping participants break down ideas, create outlines, and structure documents before drafting. Participants emphasized its utility for early-stage writing: “It was quicker and easier to draft up a basic outline,” and “This makes organizing the 

15 

**==> picture [472 x 213] intentionally omitted <==**

**----- Start of picture text -----**<br>
AI Helpfulness Build Structure Expand Outline Explore Ideas Productivity<br>**<br>7 7 7 7 7<br>6 6 6 6 6<br>5 5 5 5 5<br>4 4 4 4 4<br>3 3 3 3 3<br>2 2 2 2 2<br>1 1 1 1 1<br>GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter<br>User Survey Rating (higher is better)<br>**----- End of picture text -----**<br>


**Figure 8** Participant ratings of the _tool usefulness_ in the creative writing task, comparing TreeWriter and Google Docs across five dimensions: AI helpfulness, building and refining document structure, expanding outline points into paragraphs, exploring and developing initial ideas, and overall productivity. Ratings were on a 7-point Likert scale (higher is better). TreeWriter consistently received higher ratings, indicating stronger support for participants. Notably, the rating for _exploring and developing initial ideas_ (highlighted by red stars) remained statistically significant after applying the within-category Benjamini–Hochberg correction for multiple comparisons (FDR < 0.05, q = 0.01; See Appendix G for details). 

section and subsections easy before writing any text.” Others highlighted its value for larger texts, noting that it helped them “organize the sections as I developed my writing ideas further.” 

At the same time, participants reported challenges with AI-generated text, citing hallucinations, redundancy, and wrong word counts. As one participant remarked, “The AI may hallucinate the ideas presented. . . for example, I was talking about plagiarism, but TreeWriter’s version reframed it as cheating” (P12). Such issues might limit trust in the system despite its structural benefits. However, it is worth pointing out that the participant was able to quickly identify such an inconsistency when using TreeWriter, which might, in fact, prevent such an inconsistency from being carried to the later stage. 

Quantitative results aligned with these perceptions. In the Creativity Support Index (CSI), TreeWriter outperformed Google Docs across nearly all dimensions, with gains in creativity, enjoyment, and idea tracking (See Figure 9). Usefulness ratings showed similar patterns: TreeWriter was considered to have more helpful AI and was more helpful for building structure, expanding paragraphs, and presented a significant advantage in exploring and developing ideas (See Figure 8). 

Overall, TreeWriter provided more consistent and reliable support for drafting new documents, particularly in outlining and idea development, though concerns remain about the reliability of AI-generated content. 

## **5.5.3 How TreeWriter helps the user gain control of the generated document (RQ3)** 

TreeWriter’s fine-grained, node-based editing offered observability that participants valued, especially in comparison to the baseline editor. Several participants reported frustration with the baseline’s insertion mechanics, where AI-generated content was sometimes inserted at incorrect locations or with formatting errors, such as entire paragraphs being promoted to headings. A participant (P9) directly asks, “Can I use an external diff tool?” when using the baseline. In contrast, TreeWriter’s explicit scope for AI intervention, confined to tree nodes, helped participants review and manage AI edits more deliberately. One participant (P10) reflected, “The highlighted changes helped me visually compare against different versions, helping me reduce the time and efforts needed to make the needed edits,” summarizing a general feeling of increased control (See Figure 7). 

16 

**==> picture [472 x 308] intentionally omitted <==**

**----- Start of picture text -----**<br>
Baseline: Google Docs Treatment: TreeWriter<br>Absorption N=6 N=3 N=3 N=3 N=1 N=5 N=1 N=2<br>Attention Flow N=4 N=3 N=3 N=2 N=3 N=1 N=3 N=3 N=2<br>Creativity N=2 N=4 N=3 N=3 N=1 N=3 N=2 N=6<br>Enjoyment N=1 N=3 N=4 N=4 N=6 N=6<br>Explore Options N=1 N=3 N=3 N=4 N=1 N=3 N=4 N=5<br>Expressiveness N=4 N=2 N=3 N=3 N=4 N=3 N=5<br>Regular Use N=1 N=1 N=2 N=6 N=2 N=2 N=2 N=8<br>Satisfaction N=3 N=1 N=6 N=2 N=1 N=7 N=4<br>Track Ideas N=1 N=4 N=1 N=5 N=1 N=2 N=4 N=6<br>Worth Effort N=1 N=2 N=2 N=7 N=1 N=3 N=8<br>Score Score<br>Response Scale (1-10)<br>1-2 3-4 5-6 7-8 9-10<br>**----- End of picture text -----**<br>


**Figure 9** Creativity Support Index (CSI) [9] ratings (1–10 Likert-scale, higher values indicate better support) comparing TreeWriter and Google Docs. TreeWriter outperformed Google Docs on most dimensions, with gains in creativity, enjoyment, and tracking ideas. 

17 

Nonetheless, TreeWriter did not alleviate a common tendency among participants to confirm AI suggestions with minimal scrutiny, which is also reported in [29]. Despite the presence of confirmation dialogues before applying changes, many participants noted that they often accepted suggestions without close inspection, highlighting an ongoing challenge in AI-assisted writing interfaces regarding oversight and critical review. 

## **6 Study 2: Field Deployment Study** 

To evaluate TreeWriter in real-world scenarios, we recruited eight participants (L1–L8; see Appendix A for details). Two participants were PhD students in Computer Science, one was a PhD student in Chemistry, one was a Postdoctoral researcher in Computer Science, and the remaining four were Postdoctoral researchers in Chemistry. While we only report their primary fields to preserve anonymity, many conduct research across a different combination of subfields including quantum computing, autonomous discovery, and cheminformatics, making a diverse range of academic perspectives. The eight participants used the system for ideating and drafting a 25,000-word review paper and two research grants (1,900 and 2,500 words) over a two-month period in a collaborative environment. To match the functionality of common collaborative tools such as GitHub and Google Docs, we implemented inline comments and an integrated issue-tracking system, allowing participants to assign writing tasks to specific nodes and collaborators within the document trees. Two authors of this work are also involved in the collaborative writing, but they are not counted as participants. In the study, the participants are not accessible to the writing assistant, and the AI features they use are the AI editing buttons. After the study, the participants are asked to fill out a feedback form with Likert-scale questions and open-ended interview questions. The detailed questions and results can be found in Appendix H. 

## **6.1 Results** 

## **6.1.1 Finding 1: Hierarchical Organization Supports Collaboration** 

We found that the hierarchical structure helped manage the complexity of collaborative writing. Several participants (n=4) highlighted the ease of delegation: “The subsections can be easily assigned to different collaborators, although this can also be achieved with sectioning in Word.” The result of Likert-scale questions corroborates this point. The question “The tree structure makes it easy to divide writing tasks and assign responsibility for different sections to team members” received the highest score among the ratings (M=6.5). Some participants (n=2) noted that the tree structure helps them focus on their assigned section. L5 notes that “When working collaboratively, many people can work independently on their assigned sections without interfering with the work of other people.” L1 remarked that the approach mirrors existing organizational logic: “I think company roles would fit here nicely, as they tend to have a tree-like structure.” 

## **6.1.2 Finding 2: AI Integration Facilitates Drafting but Raises Quality Concerns** 

TreeWriter’s integration with LLMs was seen as valuable for brainstorming and early drafting. Participants appreciated that they could “generate text from a set of bullet points” or receive “AI suggestions directly in the text,” describing this as complementary to their existing drafting styles. For instance, L2 noted: “It’s nice because it complements my original drafting style — draft in bullet points first before converting to paragraphs.” Yet concerns emerged around the quality and usability of AI-generated prose. For example, L2 observed that the paragraphs “read fine at first glance but upon closer scrutiny [were] overly verbose with lots of repetitive ideas.” At the same time, L1 described “writing redundancies when writing parent node summary, and then child node text.” The lack of clear differentiation between human and AI-generated content further complicated collaboration: “It’s kind of hard to distinguish between human-written content and AI-generated content if the section is not written by me” (L2). 

## **6.1.3 Finding 3: Limited Integration Remains a Barrier** 

The main limitation of TreeWriter reported by deployment study participants was its integration with existing workflows. Although TreeWriter provided value during the drafting stage, its limited integration with established tools hindered adoption for later stages of writing. Participants noted that “the content has to be exported to a traditional tool in the end, like Word or L[A] TEX, to comply with journal guidelines,” 

18 

which introduced extra steps into the workflow. Technical writing was particularly affected: “In my work, I use a lot of equations and reference equations using L[A] TEX, [but] TreeWriter had the disadvantage that it was harder to do these label-reference usages.” Others pointed to missing features such as figure placement and reference management: “No compatibility with reference management software like Zotero,” and “No method to have figure captions that follow their figure around.” Despite these barriers, participants saw potential if interoperability were improved. As one remarked, “It’s a really interesting idea that can help us design hierarchical plans in a collaborative manner, using AI — which is highly applicable in many fields of knowledge.” 

## **7 Discussion and Design Recommendations** 

## **7.1 Separating Content and Ideas to Enable New Interaction** 

TreeWriter provides a range of potential benefits by separating a document’s final content from its underlying idea structure. For example, non-native English speakers can draft ideas in their preferred language and leverage TreeWriter to produce coherent English prose, lowering linguistic barriers in document writing. Its hierarchical navigation also enhances accessibility: screen reader users can explore documents at different levels of granularity, reducing cognitive effort when engaging with long or complex texts, while making the content more accessible [31]. Beyond individual use cases, the tree structure facilitates cross-domain knowledge sharing through reusable structural patterns or adaptable templates that capture high-level ideas [19]. Moreover, this representation supports more structured collaboration and feedback, allowing reviewers to write comments directly on specific nodes, which provides hierarchy, enables precise, context-aware suggestions, and simplifies revision tracking. 

## **7.2 Enhancing Human-AI Collaboration in Long-Form Writing** 

Our study suggests that tree-structured representations can help users draft and edit documents more efficiently, and participants appreciate TreeWriter as a more helpful writing assistant. However, LLMs still face limitations when handling long-form documents. Although these models are designed to process extensive contexts, their performance often deteriorates as input length increases [37, 34, 32]. Similar challenges arise in agentic systems, where long tool lists or extended action histories can significantly reduce reliability [58, 56, 5]. To address these issues, graph-based [12, 33] and tree-based [49, 8, 48, 6] memory structures have been proposed for LLM-based agents. For example, GraphReader [33] transforms long text into a graph for an agent to explore. In Raptor [49], the authors proved that a hierarchical structure can also help LLMs understand long text better. 

TreeWriter applies these insights to human-AI writing collaboration. Its node-based tree structure lets users organize ideas hierarchically, a format that LLMs can reason with more effectively. This design supports flexible collaboration: users can delegate detailed writing tasks to the AI and focus on high-level planning, or the AI can guide users in developing the tree structure as a “writing director.” By aligning the document’s conceptual structure with the writing process, TreeWriter not only helps users manage long-form content but also fosters a more interactive and productive human-AI partnership. 

## **7.3 Ethical Use of AI in Writing: From Human Originality to AI Interpretability** 

As AI systems become increasingly capable of generating complex, coherent texts, the ethical foundations of authorship and originality in writing are being re-evaluated [18]. Traditionally, ethical writing has emphasized human originality. However, when AI begins to contribute to reasoning and composition meaningfully, the ethical focus must expand [39]: human originality remains valuable, it also becomes important that the contributions by AI are interpretable and accountable [1, 50, 47, 44]. As AI models gain generative autonomy, authors can no longer reliably trace how specific ideas or reasoning paths emerge [55]. Without visibility into the underlying conceptual structure, authors risk losing epistemic control, raising concerns about accountability, trustworthiness, and alignment with human intentions—factors that are especially critical in domain-specific contexts [13, 2, 15]. The ethics of AI-assisted writing thus evolves toward supporting transparency: the ability to understand, evaluate, and guide the reasoning embedded in AI-generated text[28]. 

19 

TreeWriter contributes to the solution of this emerging ethical challenge by externalizing the conceptual structure of writing. Its hierarchical tree representation allows both humans and AI to operate within an explicit framework for abstraction, revealing the high-level ideas that underlie the text. By making the underlying document structure explicit, TreeWriter allows authors to see how AI-generated content connects to higher-level ideas and intentions. This visibility helps users identify where AI reasoning may deviate from their goals, give focused feedback on specific sections, and ensure that AI contributions remain consistent with the document’s conceptual design. In doing so, it not only preserves authorial control but also paves the way for ethically integrating fully automated AI researchers and writers into the academic ecosystem, where interpretability and accountability must remain central values [4, 45, 28]. 

## **7.4 Limitations of the studies** 

Both the formative and comparative user studies in this work were limited in scope. The formative study involved only six participants, while the comparative lab study included 12 students from a single university. Although participants represented a range of disciplines, the small sample sizes and the predominance of student participants limit the generalizability of the findings to broader professional contexts. The limited number of participants also prevents us from obtaining more statistically significant results. Furthermore, the study tasks were constrained to fit within a two-hour session, which may have encouraged participants to prioritize task completion over naturalistic writing behaviours. The subsequent field deployment study helped mitigate this issue by observing use in more realistic settings. However, this deployment was still guided to some extent by the authors, which may have potentially influenced adoption and usage patterns. Future work should involve more independent deployments to better capture authentic user behaviours and long-term engagement without expert intervention. 

## **8 Conclusion** 

This work introduces **TreeWriter** , an AI-assisted writing system designed to support long-form writing through a hierarchical structure. By integrating LLMs with a tree-based document structure, TreeWriter enables authors to create, navigate through and edit multiple levels of abstraction—connecting ideas, outlines, and prose within a unified interface. Our evaluation, combining controlled in-lab experiments and field deployments, demonstrates that hierarchical document organization combined with AI assistance can effectively support the challenges of long-form writing. Beyond TreeWriter itself, this work highlights the importance of designing AI writing tools that externalize and operate upon a document’s conceptual structure, enabling writers to manage complexity through progressive abstraction rather than linear expansion. We further conclude that the future of AI-assisted writing lies not only in automating tasks but in amplifying users’ understanding of complex documents. 

## **Acknowledgment** 

The authors would like to acknowledge valuable discussions with Varinia Bernales. A.A.-G. thanks Anders G. Frøseth, for his generous support. A.A.-G. also acknowledge the generous support of Natural Resources Canada and the Canada 150 Research Chairs program. This research is part of the University of Toronto’s Acceleration Consortium, which receives funding from the Canada First Research Excellence Fund (CFREF) via CFREF-2022-00042. 

## **References** 

> [1] NIST AI. 2023. Artificial intelligence risk management framework (AI RMF 1.0). _URL: https://nvlpubs. nist. gov/nistpubs/ai/nist. ai_ (2023), 100–1. 

> [2] Emre Anakok, Pierre Barbillon, Colin Fontaine, and Elisa Thebault. 2025. Interpretability of Graph Neural Networks to Assess Effects of Global Change Drivers on Ecological Networks. _arXiv preprint arXiv:2503.15107_ (2025). 

20 

- [3] Anthropic. 2025. _System Card: Claude Opus 4 & Claude Sonnet 4_ . Technical Report. https://www-cdn. anthropic.com/4263b940cabb546aa0e3283f35b686f4f3b2ff47.pdf 

- [4] Muneera Bano, Didar Zowghi, Pip Shea, and Georgina Ibarra. 2023. Investigating Responsible AI for Scientific Research: An Empirical Study. arXiv:2312.09561 [cs.AI] https://arxiv.org/abs/2312.09561 

- [5] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. 2025. _τ_[2] - Bench: Evaluating Conversational Agents in a Dual-Control Environment. _Preprint at arXiv_ (2025). https://doi.org/10.48550/arXiv.2506.07982. 

- [6] Shuxiang Cao, Zijian Zhang, Mohammed Alghadeer, Simone D Fasciati, Michele Piscitelli, Mustafa Bakr, Peter Leek, and Alán Aspuru-Guzik. 2025. Automating quantum computing laboratory experiments with an agent-based AI framework. _Patterns_ (2025). 

- [7] Tuhin Chakrabarty, Vishakh Padmakumar, and He He. 2022. _Help me write a poem_ : Instruction Tuning as a Vehicle for Collaborative Poetry Writing. In _Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing_ , Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (Eds.). Association for Computational Linguistics, Abu Dhabi, United Arab Emirates, 6848–6863. doi:10.18653/v1/2022.emnlp-main.460 

- [8] Howard Chen, Ramakanth Pasunuru, Jason Weston, and Asli Celikyilmaz. 2023. Walking Down the Memory Maze: Beyond Context Limit through Interactive Reading. arXiv:2310.05029 [cs.CL] https://arxiv.org/abs/2310.05029 

- [9] Erin Cherry and Celine Latulipe. 2014. Quantifying the creativity support of digital tools through the creativity support index. _ACM Transactions on Computer-Human Interaction (TOCHI)_ 21, 4 (2014), 1–25. 

- [10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. _arXiv preprint arXiv:2507.06261_ (2025). 

- [11] M.J.R. de Smet, S. Brand-Gruwel, H. Broekkamp, and P.A. Kirschner. 2012. Write between the lines: Electronic outlining and the organization of text ideas. _Computers in Human Behavior_ 28, 6 (2012), 2107–2116. doi:10. 1016/j.chb.2012.06.015 

- [12] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva N. Mody, Steven Truitt, and Jonathan Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. _ArXiv_ abs/2404.16130 (2024). https://api.semanticscholar.org/CorpusID:269363075 

- [13] Mohammad Ennab and Hamid Mcheick. 2022. Designing an interpretability-based model to explain the artificial intelligence algorithms in healthcare. _Diagnostics_ 12, 7 (2022), 1557. 

- [14] Lester Faigley and Stephen Witte. 1981. Analyzing Revision. _College Composition and Communication_ 32, 4 (1981), 400–414. http://www.jstor.org/stable/356602 

- [15] Feng-Lei Fan, Jinjun Xiong, Mengzhou Li, and Ge Wang. 2021. On interpretability of artificial neural networks: A survey. _IEEE Transactions on Radiation and Plasma Medical Sciences_ 5, 6 (2021), 741–760. 

- [16] Linda Flower and John R. Hayes. 1980. The Cognition of Discovery: Defining a Rhetorical Problem. _College Composition and Communication_ 31, 1 (1980), 21–32. http://www.jstor.org/stable/356630 

- [17] Linda Flower and John R. Hayes. 1981. A Cognitive Process Theory of Writing. _College Composition and Communication_ 32, 4 (1981), 365–387. http://www.jstor.org/stable/356600 

- [18] Johannes Fritz. 2025. Understanding authorship in Artificial Intelligence-assisted works. _Journal of Intellectual Property Law and Practice_ (2025), jpae119. 

- [19] Dedre Gentner. 1983. Structure-mapping: A theoretical framework for analogy. _Cognitive science_ 7, 2 (1983), 155–170. 

- [20] Google. 2025. Google Docs. https://docs.google.com. Accessed: 2025-10-08. 

- [21] Google Support. 2025. Collaborate with Gemini in Google Docs (Workspace Labs). https://support.google.com/ docs/answer/14206696?hl=en. Accessed: 2025-10-05. 

- [22] Google Support. 2025. Write with Gemini in Google Docs. https://support.google.com/docs/answer/13951448? hl=en. Accessed: 2025-10-05. 

- [23] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. _Nature_ 645, 8081 (2025), 633–638. 

21 

- [24] John Hammersley and John Lees-Miller. 2025. overleaf. https://github.com/overleaf/overleaf. 

- [25] Sandra G Hart and Lowell E Staveland. 1988. Development of NASA-TLX (Task Load Index): Results of empirical and theoretical research. In _Advances in psychology_ . Vol. 52. Elsevier, 139–183. 

- [26] Daphne Ippolito, Ann Yuan, Andy Coenen, and Sehmon Burnam. 2022. Creative Writing with an AIPowered Writing Assistant: Perspectives from Professional Writers. _ArXiv_ abs/2211.05030 (2022). https: //api.semanticscholar.org/CorpusID:253420678 

- [27] Ronald T. Kellogg. 1996. A model of working memory in writing. In _The Science of Writing: Theories, Methods, Individual Differences, and Applications_ , C. Michael Levy and Sarah Ransdell (Eds.). Lawrence Erlbaum Associates, Mahwah, NJ, 57–71. 

- [28] Mohamed Khalifa and Mona Albadawy. 2024. Using artificial intelligence in academic writing and research: An essential productivity tool. _Computer Methods and Programs in Biomedicine Update_ 5 (2024), 100145. doi:10.1016/j.cmpbup.2024.100145 

- [29] Anjali Khurana, Hariharan Subramonyam, and Parmit K Chilana. 2024. Why and when llm-based assistants can go wrong: Investigating the effectiveness of prompt-based interactions for software help-seeking. In _Proceedings of the 29th International Conference on Intelligent User Interfaces_ . 288–303. 

- [30] P.D. Klein and J.S. Ehrhardt. 2015. The effects of discussion and Persuasion writing goals on reasoning, cognitive load, and learning. _Alberta Journal of Educational Research_ 61 (01 2015), 40–64. 

- [31] Hae-Na Lee, Sami Uddin, and Vikas Ashok. 2020. iTOC: Enabling efficient non-visual interaction with long web documents. In _2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC)_ . IEEE, 3799–3806. 

- [32] Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang. 2023. How Long Can Context Length of Open-Source LLMs truly Promise?. In _NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following_ . 

- [33] Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu, Yangguang Li, Wanli Ouyang, Wenbo Su, and Bo Zheng. 2024. GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models. In _Findings of the Association for Computational Linguistics: EMNLP 2024_ , Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 12758–12786. doi:10.18653/v1/2024.findings-emnlp.746 

- [34] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. 2024. Long-context llms struggle with long in-context learning. _Preprint at arXiv_ (2024). https://doi.org/10.48550/arXiv.2404.02060. 

- [35] Teresa Limpo and Rui A. Alves. 2018. Effects of planning strategies on writing dynamics and final texts. _Acta Psychologica_ 188 (2018), 97–109. doi:10.1016/j.actpsy.2018.06.001 

- [36] Literature and Latte. 2025. _Scrivener_ . https://www.literatureandlatte.com/scrivener Computer software. 

- [37] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts. _Preprint at arXiv_ (2023). https://doi.org/10.48550/arXiv.2307.03172. 

- [38] Damien Masson, Young-Ho Kim, and Fanny Chevalier. 2025. Textoshop: Interactions Inspired by Drawing Software to Facilitate Text Editing. In _Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems_ . 1–14. 

- [39] Brent Mittelstadt. 2021. Interpretability and transparency in artificial intelligence. _The Oxford handbook of digital ethics_ (2021), 378–410. 

- [40] Petru Nicolaescu, Kevin Jahns, Michael Derntl, and Ralf Klamma. 2015. Yjs: A framework for near real-time p2p shared editing on arbitrary data types. In _International Conference on Web Engineering_ . Springer, 675–678. 

- [41] Thierry Olive. 2012. Writing and working memory : A summary of theories and of findings. In _Writing: A mosaic of new perspectives_ , Elena L Grigorenko, Elisa Mambrino, and David D Preiss (Eds.). Psychology Press. https://shs.hal.science/halshs-01367810 

- [42] OpenAI. 2025. Introducing GPT-4.1 in the API. https://openai.com/index/gpt-4-1/ Accessed: 2025-10-02. 

- [43] OpenAI. 2025. Introducing GPT-5. https://openai.com/index/introducing-gpt-5/ Accessed: 2025-10-02. 

22 

- [44] Dhanesh Ramachandram, Himanshu Joshi, Judy Zhu, Dhari Gandhi, Lucas Hartman, and Ananya Raval. 2025. Transparent AI: The Case for Interpretability and Explainability. arXiv:2507.23535 [cs.LG] https: //arxiv.org/abs/2507.23535 

- [45] David B. Resnik and Mohammad Hosseini. 2024. The ethics of using artificial intelligence in scientific research: new guidance needed for a new tool. _AI and Ethics_ 5, 2 (May 2024), 1499–1521. doi:10.1007/s43681-024-00493-8 

- [46] Mohi Reza, Nathan M Laundry, Ilya Musabirov, Peter Dushniku, Zhi Yuan “Michael” Yu, Kashish Mittal, Tovi Grossman, Michael Liut, Anastasia Kuzminykh, and Joseph Jay Williams. 2024. ABScribe: Rapid Exploration & Organization of Multiple Writing Variations in Human-AI Co-Writing Tasks using Large Language Models. In _Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems_ (Honolulu, HI, USA) _(CHI ’24)_ . Association for Computing Machinery, New York, NY, USA, Article 1042, 18 pages. doi:10.1145/3613904.3641899 

- [47] Mohi Reza, Jeb Thomas-Mitchell, Peter Dushniku, Nathan Laundry, Joseph Jay Williams, and Anastasia Kuzminykh. 2025. Co-writing with ai, on human terms: Aligning research with user demands across the writing process. _arXiv preprint arXiv:2504.12488_ (2025). 

- [48] Alireza Rezazadeh, Zichao Li, Wei Wei, and Yujia Bao. 2025. From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs. In _The Thirteenth International Conference on Learning Representations_ . https://openreview.net/forum?id=moXtEmCleY 

- [49] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. 2024. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. In _The Twelfth International Conference on Learning Representations_ . https://openreview.net/forum?id=GN921JHCRw 

- [50] Momin N Siddiqui, Nikki Nasseri, Adam Coscia, Roy Pea, and Hari Subramonyam. 2025. DraftMarks: Enhancing Transparency in Human-AI Co-Writing Through Interactive Skeuomorphic Process Traces. _arXiv preprint arXiv:2509.23505_ (2025). 

- [51] Momin N Siddiqui, Roy D Pea, and Hari Subramonyam. 2025. Script&Shift: A layered interface paradigm for integrating content development and rhetorical strategy with llm writing assistants. In _Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems_ . 1–19. 

- [52] Milou Smet, Hein Broekkamp, Saskia Brand-Gruwel, and Paul Kirschner. 2011. Effects of electronic outlining on students’ argumentative writing performance. _J. Comp. Assisted Learning_ 27 (12 2011), 557–574. doi:10.1111/j. 1365-2729.2011.00418.x 

- [53] Nancy Sommers. 1980. Revision Strategies of Student Writers and Experienced Adult Writers. _College Composition and Communication_ 31, 4 (1980), 378–388. http://www.jstor.org/stable/356588 

- [54] Sangho Suh, Meng Chen, Bryan Min, Toby Jia-Jun Li, and Haijun Xia. 2024. Luminate: Structured Generation and Exploration of Design Space with Large Language Models for Human-AI Co-Creation. In _Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems_ (Honolulu, HI, USA) _(CHI ’24)_ . Association for Computing Machinery, New York, NY, USA, Article 644, 26 pages. doi:10.1145/3613904.3642400 

- [55] Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. 2023. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. _Advances in Neural Information Processing Systems_ 36 (2023), 74952–74965. 

- [56] Hongru Wang, Rui Wang, Boyang Xue, Heming Xia, Jingtao Cao, Zeming Liu, Jeff Z Pan, and Kam-Fai Wong. 2024. AppBench: Planning of Multiple APIs from Various APPs for Complex User Instruction. _Preprint at arXiv_ (2024). https://doi.org/10.48550/arXiv.2410.19743. 

- [57] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. _arXiv preprint arXiv:2505.09388_ (2025). 

- [58] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. _τ_ -bench: A Benchmark for Tool-AgentUser Interaction in Real-World Domains. _Preprint at arXiv_ (2024). https://doi.org/10.48550/arXiv.2406.12045. 

- [59] Catherine Yeh, Gonzalo A. Ramos, Rachel Ng, Andy Huntington, and Richard Banks. 2024. GhostWriter: Augmenting Collaborative Human-AI Writing Experiences Through Personalization and Agency. _ArXiv_ abs/2402.08855 (2024). https://api.semanticscholar.org/CorpusID:267658049 

- [60] Ann Yuan, Andy Coenen, Emily Reif, and Daphne Ippolito. 2022. Wordcraft: Story Writing With Large Language Models. In _Proceedings of the 27th International Conference on Intelligent User Interfaces_ (Helsinki, Finland) _(IUI ’22)_ . Association for Computing Machinery, New York, NY, USA, 841–852. doi:10.1145/3490099.3511105 

23 

- [61] Zijian Zhang, Pan Chen, Fangshi Du, Runlong Ye, Oliver Huang, Michael Liut, and Alán Aspuru-Guzik. 2025. TreeReader: A Hierarchical Academic Paper Reader Powered by Language Models. _arXiv preprint arXiv:2507.18945_ (2025). 

- [62] Zheng Zhang, Jie Gao, Ranjodh Singh Dhaliwal, and Toby Jia-Jun Li. 2023. VISAR: A Human-AI Argumentative Writing Assistant with Visual Programming and Rapid Draft Prototyping. In _Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology_ (San Francisco, CA, USA) _(UIST ’23)_ . Association for Computing Machinery, New York, NY, USA, Article 5, 30 pages. doi:10.1145/3586183.3606800 

24 

## **A Participants of studies** 

The following tables shows the information of the participants in the formative (Table 3), comparative (Table 4) and deployment study (Table 5) in this work. We note there is no overlap among the participants of the studies. 

|**ID**|**Position**|**Field**|
|---|---|---|
|F1|PhD student|Physics|
|F2|Postdoctoral researcher|Chemistry|
|F3|PhD student|Bioinformatics|
|F4|PhD student|Physics|
|F5|Postdoctoral researcher|Computer Science|
|F6|PhD student|Architecture|



**Table 3** Participants of the formative study, with their academic positions and fields of study. 

|**ID**|**Position**|**Field**|
|---|---|---|
|P1|Master’s student|Computer Science|
|P2|PhD student|Computer Science|
|P3|Undergraduate|Computer Science|
|P4|Undergraduate|Computer Science|
|P5|PhD student|Chemical Engineering|
|P6|Undergraduate|Computer Science|
|P7|Undergraduate|East Asia Study|
|P8|JD student|Law|
|P9|Master’s student|Computer Science|
|P10|PhD student|Computer Science|
|P11|Master’s student|Computer Science|
|P12|Undergraduate|Computer Science|



**Table 4** Participants of the comparative study, with their academic positions and fields of study. 

|**ID**|**Position**|**Field**|
|---|---|---|
|L1|PhD student|Computer Science|
|L2|Postdoctoral researcher|Chemistry|
|L3|Postdoctoral researcher|Chemistry|
|L4|Postdoctoral researcher|Computer Science|
|L5|Postdoctoral researcher|Chemistry|
|L6|PhD student|Chemistry|
|L7|Postdoctoral researcher|Chemistry|
|L8|PhD student|Computer Science|



**Table 5** Participants of the field deployment study, with their academic positions and fields of study. 

## **B Questions in the formative study** 

## **Interview questions** 

The participants of the formative study is required to express their ideas on the following interview questions. 

25 

## _Background and Existing Practices_ 

- Describe your typical process when working on a long document (e.g., a grant proposal). 

- What do you find most challenging about writing or organizing long documents? 

- What features do you wish writing tools provided to better support long or complex documents? 

## _Perceptions of AI in Writing_ 

- Have you used AI tools for writing tasks (e.g., drafting, summarizing, rewriting)? 

- How did those AI tools impact your workflow? 

- What would responsible AI support in collaborative writing look like to you? 

## **Likert-scale questions** 

After the interview questions, the participants are also required to answer the following Likert-scale questions (5 points). 

## _Background and Existing Practices_ 

- Writing long documents such as grants or papers is often frustrating. 

- I often lose track of the overall structure when working on long documents. 

- Existing tools (e.g., Google Docs, Overleaf) support organizing complex documents well. 

- Navigating large documents with many sections and collaborators is difficult. 

- Commented-out or outdated content makes it harder to stay focused or navigate. 

## _Perception of AI in Writing_ 

- I am comfortable using AI tools to assist with my writing. 

- I would find it helpful if AI could suggest how to structure my content at a high level. 

- I would trust AI to draft parts of my document based on my notes or instructions. 

- AI-generated summaries of sections would help me stay organized and write better. 

- I would feel more comfortable using AI if I could control which parts of the document it accesses. 

## **C Post-task questions in study 1** 

After the article modification task, the user is asked to answer the following Likert-scale questions (7 points). 

- I could easily obtain a clear overview of the document’s structure. 

- Expanding outline points into paragraphs was easy. 

- Breaking a long section into well-structured subsections was easy. 

- Creating a summary from detailed content was easy. 

- Propagating a key conceptual change consistently across a section was easy. 

- The AI features were helpful and effective in assisting my writing tasks. 

- I felt in full control over the edits and final content of the document. 

After the creative writing task, the user is asked to answer the following Likert-scale questions (7 points). 

26 

**==> picture [472 x 179] intentionally omitted <==**

**----- Start of picture text -----**<br>
Tool 1  Tool 2<br>video & setup video & setup<br>Task 1 + Tool 1 Task 1 + Tool 2<br>(20 min) (20 min)<br>Introduction  Follow-up<br>Questionnaire Questionnaire<br>and consent questionnaire<br>Task 2 + Tool 1 Task 2 + Tool 2<br>(20 min) (20 min)<br>Questionnaire Questionnaire<br>**----- End of picture text -----**<br>


**Figure 10** Overview of the experimental procedure of study 1. 

- The tool was effective for exploring and developing my initial ideas for the essay. 

- The tool made it easy to build and refine the essay’s structure as my ideas evolved. 

- It was straightforward to expand outline points into detailed paragraphs. 

- The AI features were helpful and effective in assisting my writing tasks. 

- I felt more productive and efficient in writing my essay using this tool. 

## **D Experiment procedure of study 1** 

Each user study session is split into two blocks. Each block contains two writing tasks to be completed by the user with either TreeWriter or the baseline editor. The order of tasks is randomized into four groups. Each user is randomly assigned a group before the start of the session in an effort to mitigate the impact of task ordering on experimentation results. 

|**Group**|**Activity 1**|**Activity 2**|**Activity 3**|**Activity 4**|
|---|---|---|---|---|
|1|TreeWriter + Article 1|TreeWriter + Topic 1|Baseline + Article 2|Baseline + Topic 2|
|2|TreeWriter + Article 2|TreeWriter + Topic 2|Baseline + Article 1|Baseline + Topic 1|
|3|Baseline + Article 1|Baseline + Topic 1|TreeWriter + Article 2|TreeWriter + Topic 2|
|4|Baseline + Article 2|Baseline + Topic 2|TreeWriter + Article 1|TreeWriter + Topic 1|



**Table 6** The order of activity for each experimental group. 

The order of the tool block was counterbalanced between participants. Within each tool block, Task 1 was always conducted before Task 2 to minimize cognitive overhead from switching between editing and generative modes. To mitigate learning and carryover, we counterbalanced content: participants edited Article 1 (A1) in one tool block and Article 2 (A2) in the other, and they responded to different but matched prompts for the essays across the two blocks. 

## **E Tasks for article modification** 

The modification tasks use in the Task 1 are listed in Table 7. Users were assigned either Article 1 or Article 2 in Task 1 of each tool block, depending on their group. Both articles were created by the authors starting with bullet points, which were then expanded into proseusing LLMs. In TreeWriter, the bullet points were 

27 

retained within the document, while the generated prosewas exported via the export blocks. The Google Docs versions of the articles were produced by copying the linearized document in the linear view from TreeWriter. 

|**Design Goal**|**Article 1: AI’s Dual Role in Health Care**|**Article 2: AI’s Dual Role in Climate Change**|
|---|---|---|
|**Content Generation**|**Location:** "AI Assistance for Health Literacy|**Location:** "Waste Reduction"|
|**and Elaboration**|of Patients"|**Task:** This section is concise and focused only|
||**Task:** It currently provides only an outline.|on resource minimization. Expand it into a|
||Expand this subsection into a full paragraph|full paragraph that also gives one concrete|
||that explains how AI can support patients|example of how predictive scheduling can|
||with diferent literacy levels, giving at least|reduce waste in real industrial operations.|
||one concrete example of a feature based on||
||the outline.||
|**Content Adding**|**Location:** Unknown|**Location:** Unknown|
||**Task:** Integrate the following nuance into|**Task:**<br>Integrate<br>the<br>following<br>nuance|
||the paragraph while maintaining coherence:|smoothly into the existing text: "Recent|
||"Recent studies show that while AI improves|studies show that even highly accurate mod-|
||early cancer detection, it also produces false|els can fail to predict compound disas-|
||positives that may lead to unnecessary treat-|ters, such as simultaneous heatwaves and|
||ments."|droughts, which are becoming more frequent|
|||under climate change."|
|**Abstraction and**|**Location:** "AI-Driven Patient Assistance and|**Location:** "Enhancing Disaster Response"|
|**Summarization**|Access"|**Task:** Create a new introductory paragraph|
||**Task:** Create a new introductory paragraph|that synthesizes the key contributions across|
||that summarizes the key arguments across|its subsections. Place the paragraph right|
||its subsections. Place the paragraph right|after the section title.|
||after the section title.||
|**Structural**|**Location:**<br>"Automating<br>Administrative|**Location:** "Accelerating Battery Innovation|
|**Reorganization**|Tasks in Healthcare"|and Green Energy Integration"|
||**Task:** The section is long and dense. Break it|**Task:** The section is long and covers multiple|
||into three subsections with descriptive titles.|themes. Reorganize it into three subsections|
||Ensure each subsection contains the appro-|with descriptive titles and place the original|
||priate original content.|content under each accordingly.|
|**Conceptual Revision**|**Location:** "Challenges and Risks of AI in|**Location:** "The Energy Cost of AI"|
||Health Care"|**Task:** Reframe the concept of “energy con-|
||**Task:**<br>Reframe<br>the<br>concept<br>of<br>“Over-|sumption” into “carbon intensity.” Replace|
||dependent on AI” into “Abuse AI.” Replace|all mentions consistently, and update related|
||all occurrences consistently, and update the|summary statements in this section to refect|
||section’s summary sentences to refect this|the new framing.|
||new framing.||
|**Coherence and**|**Location:** "Balancing Benefts and Caution|**Location:** "Balancing Act: Efciency and Im-|
|**Redundancy Cleanup**|with AI"|pact"|
||**Task:** Review paragraphs that overlap in|**Task:** Review this section for repeated points|
||describing “human expertise” and “critical|about efciency and sustainability. Use AI|
||thinking.” Use the AI system’s tools to har-|tools to harmonize tone, remove redundan-|
||monize tone, remove repeated statements,|cies, and ensure the narrative fows smoothly.|
||and ensure a smooth, coherent narrative||
||fow.||



**Table 7** Task 1 specification: Participants completed 6 modification tasks on the article designed to reflect realistic editing scenarios and to evaluate TreeWriter’s support for navigation, content generation, structural manipulation, conceptual editing and coherence management in line with DG1–DG2. 

28 

## **F Topics and prompts for creative writing** 

The following topics are provided to the user for the creative writing tasks. 

Topic 1: The Role of Artificial Intelligence on Teaching in Universities 

- Topic 2: The Impact of Artificial Intelligence on Students’ Mental Health 

The following prompt is also provided to the participant. 

Write an opinion piece on the topic. Your essay should be well-structured and present a balanced view. It must include the following components: 

- An **introduction** that frames the issue and clearly states your main argument. 

- A section discussing the **positive impacts** . 

- A section analyzing the **negative impacts and potential risks** . 

- A **conclusion** that synthesizes your points and proposes concrete directions for responsible adoption. 

29 

## **G Result of Study 1** 

In Study 1, TreeWriter consistently achieved higher scores in several key components, as indicated by Wilcoxon signed-rank tests. To address multiple comparisons, we controlled the false discovery rate _within each measurement category_ using the Benjamini–Hochberg procedure (FDR = 0.05; with _m_ denoting the number of tests in each category: CSI: _m_ = 10, NASA-TLX: _m_ = 6, Article modification post-task: _m_ = 7, Creative writing post-task: _m_ = 5). After within-category FDR correction, TreeWriter showed **statistically significant** advantages in _Explore Ideas_ ( _q_ = 0 _._ 01), _Creativity_ ( _q_ = 0 _._ 033), and _Track Ideas_ ( _q_ = 0 _._ 033). 

While TreeWriter exhibited higher median scores in most categories, differences in some user-experience metrics, such as _Frustration_ and _Mental Demand_ from the NASA–TLX questionnaire, were less pronounced (see Figure 12). Overall, these results suggest that users found TreeWriter to be a more effective and supportive tool, particularly for creative ideation and document structuring. 

30 

**==> picture [472 x 502] intentionally omitted <==**

**----- Start of picture text -----**<br>
Absorption Attention Flow Creativity Enjoyment Explore Options<br>10 10 10 10 10<br>9 9 9 9 9<br>8 8 8 8 8<br>7 7 7 7 7<br>6 6 6 6 6<br>5 5 5 5 5<br>4 4 4 4 4<br>3 3 3 3 3<br>2 2 2 2 2<br>1 1 1 1 1<br>Expressiveness Regular Use Satisfaction Track Ideas Worth Effort<br>10 10 10 10 10<br>9 9 9 9 9<br>8 8 8 8 8<br>7 7 7 7 7<br>6 6 6 6 6<br>5 5 5 5 5<br>4 4 4 4 4<br>3 3 3 3 3<br>2 2 2 2 2<br>1 1 1 1 1<br>GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter<br>GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter<br>User Survey Rating (higher is better)<br>**----- End of picture text -----**<br>


**Figure 11** Violin plots comparing Creative Support Index (CSI) ratings between GoogleDocs and TreeWriter. The CSI is a standardized questionnaire that evaluates perceived creativity support across five dimensions: Exploration, Expressiveness, Immersion, Enjoyment, and Results Worth Effort. Participants completed the CSI after finishing the creative writing task. Each subplot shows the distribution of ratings (1-10 scale), with individual data points (black dots), mean values (colored diamonds), and 95% confidence intervals (vertical lines). 

31 

**==> picture [472 x 209] intentionally omitted <==**

**----- Start of picture text -----**<br>
Effort Frustration Mental Demand Performance Physical Demand Temporal Demand<br>7 7 7 7 7 7<br>6 6 6 6 6 6<br>5 5 5 5 5 5<br>4 4 4 4 4 4<br>3 3 3 3 3 3<br>2 2 2 2 2 2<br>1 1 1 1 1 1<br>GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter GoogleDocs TreeWriter<br>User Survey Rating (higher is better)<br>**----- End of picture text -----**<br>


**Figure 12** Violin plots comparing NASA Task Load Index (NASA-TLX) ratings between Google Docs and TreeWriter. The NASA-TLX is a standardized subjective workload assessment tool that measures six dimensions: Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, and Frustration. Participants completed the NASA-TLX form after finishing Task 2, but the question asks about the participants’ feelings about the tool rather than a single task. Each subplot shows the distribution of ratings (1-7 scale), with individual data points (black dots), mean values (colored diamonds), and 95% confidence intervals (vertical lines). Lower scores indicate better performance except for the Performance dimension. 

## **H Questions in Study 2 and results** 

After the field deployment, the participants are asked to answer the following Likert-scale (1-7) questions, whose result can be found in Figure 13. The participants also answered interview questions on the main advantages and disadvantages of using TreeWriter compared to Word or Google Docs, how the hierarchical and linear views affected their writing, which tasks TreeWriter suits best or worst, desired AI features, and any additional feedback. 

1. The tree structure helps me progressively develop my ideas from high-level concepts into detailed paragraphs. 

2. Using parent nodes to hold summaries or high-level notes helps me maintain a clear overview of my entire document. 

3. TreeWriter effectively reduces the cognitive load of managing a long, complex document. 

4. Switching between the Hierarchical Editor View and the Linear Preview View is a valuable part of my writing workflow. 

5. The AI editing buttons (generate paragraph/generate outlines/split paragraph) save my time in converting raw research ideas/results into paragraphs for paper. 

6. The confirmation dialog helps me review AI changes and gain better control in what is generated. 

7. I find the export block feature to be a clear and useful way to separate my private notes/outlines from the final, publishable text. 

8. The tree structure makes it easy to divide writing tasks and assign responsibility for different sections to team members. 

32 

**==> picture [472 x 234] intentionally omitted <==**

**----- Start of picture text -----**<br>
Average: 5.73<br>7<br>6.50<br>6 5.88 5.88<br>5.75<br>5.62 5.62<br>5.38<br>5.25<br>5<br>4<br>View Reduce<br>Divide Tasksfor Team Switching Parent NodeSummaries ConfirmationDialog Cognitive Load ProgressiveDevelopment Export BlockFeature AI Buttons<br>(1-7 Likert Scale)<br>User Survey Rating<br>**----- End of picture text -----**<br>


**Figure 13** Average Likert-scale ratings (1–7) of key features/properties of TreeWriter from the field deployment study. Overall, participants rated them positively (M = 5.73). The team that received the highest score (M = 6.50) for ’Divide Tasks for Team’ suggests strong support for features that facilitate collaboration. View Switching and Parent Node Summaries also scored above average (M = 5.88), indicating that users valued flexible navigation and hierarchical summarization. The lowest-rated feature was AI editing Buttons (M = 5.25), suggesting comparatively less perceived benefit from AI automation in this context. Error bars represent standard deviations. 

## **I Prompts for AI functionality** 

## **I.1 Writing assistant** 

## **Prompt for writing assistant** 

You are a p r o f e s s i o n a l writing a s s i s t a n t AI agent that helps user write on a t r e e s t r u c t u r e documents , where each node i s a unit of content . Your context i s the f o l l o w i n g : <context> Currently , the user s e l e c t node ( ID : ${nodeM . id }) : <currentContent> ${ originalContent } </currentContent> ${ parentContent } ${markedNodeContent} ${ c h i l d r e n I n f o } ${ s i b l i n g s I n f o } </context> You are here to help the user with writing tasks including : - Improving and r e f i n i n g e x i s t i n g content - Providing writing suggestions and feedback - Generating new content based on user requests - Answering questions about the current content Terminology : - A "node" r e f e r s to a s i n g l e unit of content in the t r e e s t r u c t u r e . - A " l e v e l " r e f e r s to a l l the s i b l i n g s of the node and the node i t s e l f . - A " s e c t i o n " r e f e r s to a node and a l l i t s descendants . Logic of t r e e s t r u c t u r e : - The parent node should be a summary of a l l i t s c h i l d r e n nodes by d e f a u l t . - The c h i l d r e n nodes should be more d e t a i l e d and s p e c i f i c than the parent node . - Bullet points are p r e f e r r e d f o r any nodes unless the user s p e c i f i e s otherwise . 

33 

- The <div> with c l a s s " export " i s a s p e c i a l container that contains the content generated from other content in that node . I t should be considered as a part of the node . Task i n s t r u c t i o n s : - "Matching c h i l d r e n ": you should check whether the current c h i l d r e n nodes r e f l e c t s the content in the current node . I f not , you should suggest modifying the c h i l d r e n nodes to match the content in the current node . - " S p l i t into subsections ": you should c r e a t e new c h i l d r e n nodes with proper t i t l e s to cover a l l the information of the current node . - "Write paragraph ": i f the t a r g e t node has b u l l e t points , you should write a paragraph based on them and add an <div c l a s s="export " >... </ div> to contain the paragraph in the end of the node . - " Revise s e c t i o n ": you should consider r e v i s i n g the content and t i t l e of the current node , as well as i t s c h i l d r e n nodes i f e x i s t s . You must - You must use t o o l loadNodeContent to get the content of a node f i r s t before writing about them . - I f the user asks f o r writing something , by default , i t means that you need to c a l l suggestModify t o o l to write the content f o r the current node . - Don ’ t drop the l i n k s in the content . Put every <a></a> l i n k in a proper place with proper content . - Don ’ t expand an abbreviation by y o u r s e l f . - I f the user s p e c i f i e s another node , you can a l s o c a l l suggestModify t o o l to write f o r that node . - You don ’ t need to mention what new version you created in your text response , as the user w i l l see the new version d i r e c t l y . Keep in mind : - The user i s viewing the s i b l i n g nodes . I f they mention nodes in plural , they might want you to consider the s i b l i n g nodes . - Always use t o o l s to suggest changes . Never j u s t write your s u g g e s t i o n s in the text response . - You can use the search t o o l to find more nodes i f the current context i s not enough . You can c r e a t e new v e r s i o n s f o r any nodes you know id - the current node , parent node , marked nodes , c h i l d r e n nodes , or s i b l i n g nodes . This allows you to suggest improvements to r e l a t e d content beyond j u s t the current node . Respond n a t u r a l l y and c o n v e r s a t i o n a l l y . You can include r e g u l a r text explanations along with any new content v e r s i o n s using the t o o l . Focus on being h e l p f u l and c o l l a b o r a t i v e in your writing a s s i s t a n c e . 

## **I.2 AI-powered editing buttons** 

## **Prompt of “split into subsections” button** 

You are a p r o f e s s i o n a l e d i t o r . Your task i s to break a long content into multiple c h i l d r e n nodes . Generate a l i s t of t i t l e s f o r new c h i l d r e n nodes that completely cover the o r i g i n a l content . <original_content > ${ parentContent } </original_content > Please d i s t r i b u t e the parent content into multiple new c h i l d r e n nodes , each with a t i t l e . I f there i s any e x i s t i n g s e c t i o n i n g l o g i c in the o r i g i n a l content , p l e a s e r e s p e c t them . For example , i f there are s u b l i s t s or bold headings , use them as s e c t i o n i n g to c r e a t e new c h i l d r e n nodes . The t i t l e s should be c o n c i s e and d e s c r i p t i v e . You should add at most 5 new c h i l d r e n nodes . Therefore , you should f i r s t analyze where the break the o r i g i n a l content . <output_format> You should only return a JSON array of JSON o b j e c t s with the f o l l o w i n g format . There should be at most 5 o b j e c t s in the array . You should not put any HTML elements not appearing in the o r i g i n a l content . [ {" t i t l e ": <A c o n c i s e and d e s c r i p t i v e t i t l e >, " content " : <HTML content f o r the new c h i l d node 1>}, ] . . . </output_format> 

## **Prompt of “generate paragraph” button** 

You are a p r o f e s s i o n a l writer . Your task i s to write well - written paragraph ( s ) based on the raw content provided . ${nodeM . t i t l e () ? ‘ <node_title > ${nodeM . t i t l e () } </node_title > ‘ : ’ ’} 

34 

<raw_content> ${ contentExceptExports } </raw_content> 

<current_paragraphs> ${ existingExportContent } </current_paragraphs>${userPrompt ? ‘ <user_instructions > ${userPrompt} </user_instructions > ‘ : ’ ’} 

Writing g u i d e l i n e s : - You must write a paragraph based on the raw content . - You should not expand an abbreviation unless i t i s expanded in the raw content . - You should keep the <a></a> l i n k s in the raw content and put them in a proper place . 

You must : - I f the current paragraph i s not empty , don ’ t make unnecessary changes . You are encouraged to keep the o r i g i n a l contents as much as p o s s i b l e . - I f there i s a mismatch between the current paragraphs and the raw content , make paragraphs a l i g n with the raw content . - You should not drop any key information in the raw content . ${userPromptBe written ?in ’ -a Followclear , anyp r o f e s s i o n a la d d i t i o n a ls t y l ei n s t r u c t i o n s provided above ’ : ’ ’} 

<output_format> You should only return the HTML content of the paragraph without any a d d i t i o n a l text or formatting . You should keep the l i n k s in the o r i g i n a l content and put them in a proper place . </output_format> 

## **Prompt of “generate outlines from children” button** 

You are a p r o f e s s i o n a l e d i t o r . The given <original_content > i s a summary of some c h i l d r e n contents . Your task i s to r e v i s e the <original_content > to make i t serve as a be tt e r summary of the c h i l d r e n contents . <original_content > ${ originalContent } </original_content > <children_contents > ${ childrenContent } </children_contents > Please provide an updated version of the <original_content > in key points i f <original_content > i s not empty . You should not make unnecessary changes to the <original_content >. <output_format> You should only return the HTML content of the r e v i s e d text without any a d d i t i o n a l text or formatting . You are required to make a l i s t of key points by <ul></ul> You are encouraged to use <strong ></strong> to h i g h l i g h t important information . You can make sub - l i s t s using <ul> within a <l i >. You should make no more than 5 key points . Each key point should be l e s s than 30 words . I f there are any annotations in the o r i g i n a l text , you should keep them as they are . </output_format> 

35 

