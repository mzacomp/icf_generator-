Enhanced ICF Generator 

 The Enhanced ICF Generator is an AI-powered web application that automatically generates Informed Consent Forms from clinical trial protocols using advanced retrieval strategies that go beyond traditional semantic similarity search. Instead of relying on basic RAG approaches, the system implements context-aware chunk classification that categorizes content into types like "planned_action," "completed_action," "risk_info," and "benefit_info" to address the critical problem of semantically similar text having different meanings in different contexts. The application uses section-specific filtering to ensure that each ICF section (Purpose, Procedures, Risks, Benefits) retrieves only relevant content types, preventing context pollution between different sections. These four AI-generated section are then fit specifically within the boilerplate template of the Informed Consent Protocol. Built with a FastAPI backend and React frontend, the system successfully processed a 251-page clinical trial protocol into 1,591 classified chunks and generated professional-quality ICF documents with complete citation tracking. The enhanced retrieval strategy specifically addresses challenges in clinical document processing where temporal distinctions (planned vs completed procedures) and domain-specific categorization are crucial for accuracy. The system includes built-in verification and judgment components with a Judge LLM that assess the quality and faithfulness of generated content against source material. Testing with real clinical trial protocols demonstrated the system's ability to extract clinically accurate information while maintaining regulatory compliance for readability standards. 





Frontend: 
<img width="1370" height="807" alt="Screenshot 2025-09-26 at 11 46 42 AM" src="https://github.com/user-attachments/assets/4a073687-69dd-4918-84b6-9a1222f6d777" />


Clinical Trial Protocol: 

<img width="796" height="818" alt="Screenshot 2025-09-26 at 12 53 49 PM" src="https://github.com/user-attachments/assets/73e9f0ae-c9ff-42d0-b113-39e012f66251" />


Generated Informed Consent Form based on the Clinical Trial Protocol: 
<img width="726" height="762" alt="Screenshot 2025-09-26 at 12 57 44 PM" src="https://github.com/user-attachments/assets/e1cf1cb7-c747-4d75-b557-1f090876c216" />

JSON Logs For Audibility: 


<img width="1013" height="447" alt="Screenshot 2025-09-26 at 1 03 18 PM" src="https://github.com/user-attachments/assets/bb4b4646-18ed-4adc-bfbb-a97f026832d7" />




