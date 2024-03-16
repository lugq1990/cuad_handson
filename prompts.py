from langchain.prompts import PromptTemplate


SUMMARY_TEMPLATE = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["input", "chat_history"],
    template=SUMMARY_TEMPLATE
)

openai_prompt = """Scenario: You are a legal professional working for a law firm that deals with various types of contracts. Your firm has recently developed an advanced content search engine that utilizes AI language model, LLM, to assist in analyzing and extracting relevant information from legal contracts.

Content: You start by providing LLM with a snippet of a legal contract. The contract excerpt contains specific clauses related to confidentiality, termination, and dispute resolution. Describe these clauses in detail, including any key terms or conditions mentioned.

Query: After providing the content snippet, you ask LLM the following question:

"Based on the provided contract excerpt, can you identify any potential issues or risks related to confidentiality, termination, or dispute resolution? Additionally, please suggest any best practices or recommendations for handling these clauses effectively."

This prompt sets the stage for a conversation where LLM can analyze the provided content (the contract snippet) and extract relevant information related to confidentiality, termination, and dispute resolution. LLM's response can include identifying potential legal risks or concerns in those clauses, along with offering best practices and recommendations for handling these critical contract elements effectively. This interaction demonstrates how the content search engine powered by LLM can be a valuable tool for legal professionals to analyze and understand complex legal contracts."""


sample ="""Scenario: You are a legal professional working for a law firm that deals with various types of contracts. Your firm has recently developed an advanced content search engine that utilizes AI language model, LLM, to assist in analyzing and extracting relevant information from legal contracts.
Content: Source: VERIZON ABS LLC, 8-K, 1/23/2020will survive termination of this Agreement and may not be waived by the Issuer or the Indenture Trustee:
(i) Valid Security Interest . This Agreement creates a valid and continuing security interest (as defined in the applicable UCC) in
the Depositor Transferred Property in favor of the Issuer, which is prior to all other Liens, other than Permitted Liens, and is enforceable
against creditors of, purchasers from and transferees and absolute assignees of the Depositor.
(ii)  Type . Each Receivable is (A) if the Receivable is not secured by the related Device, an “account” or “payment intangible,” or
(B) if the Receivable is secured by the related Device, “chattel paper,” in each case, within the meaning of the applicable UCC.
(iii)  Good Title . Immediately before the transfer and absolute assignment under this Agreement, the Depositor owns and has good
title to the Depositor Transferred Property free and clear of all Liens, other than Permitted Liens. The Depositor has received all consents
and approvals required by the terms of the Depositor Transferred Property to Grant to the Issuer its right, title and interest in the Depositor
Transferred Property, except to the extent the requirement for consent or approval is extinguished under the applicable UCC.
(iv)  Filing Financing Statements . The Depositor has caused, or will cause within ten (10) days after the Closing Date, the filing of
all appropriate financing statements in the proper filing office in the appropriate jurisdictions under applicable Law to perfect the security
interest Granted in the Depositor Transferred Property to the Issuer under this Agreement. All financing statements filed or to be filed
against the Depositor in favor of the Issuer under this Agreement describing the Depositor Transferred Property will contain a statement
to the following effect: “A purchase, absolute assignment or transfer of or grant of a security interest in any collateral described in this
financing statement will violate the rights of the Secured Parties.”
(v)  No Other Transfer, Grant or Financing Statement . Other than the security interest Granted to the Issuer under this Agreement,
the Depositor has not transferred or Granted a security interest in any of the Depositor Transferred Property. The Depositor has not
authorized the filing of and is not aware of any financing statements against the Depositor that include a description of collateral covering
any of the Depositor Transferred Property, other than financing statements relating to the security interest Granted to the Issuer
Query:Based on the provided contract excerpt, can you identify any potential issues or risks related to confidentiality, termination, or dispute resolution? Additionally, please suggest any best practices or recommendations for handling these clauses effectively.
"""

prompt_template = """"Scenario: You are a legal professional working for a law firm that deals with various types of contracts. Your firm has recently developed an advanced content search engine that utilizes AI language model, LLM, to assist in analyzing and extracting relevant information from legal contracts.

Content: {}

Query: {}
"""

open_ai_provide_prompt = """
Analyzing the risks associated with Clause [X] from the previous legal contract. Please consider the legal implications, potential financial risks, and compliance considerations. Propose any language modifications or additional provisions that could mitigate identified risks. Provide a concise summary of key insights and considerations. Include relevant industry standards in your analysis.
"""