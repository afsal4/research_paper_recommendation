import streamlit as st 
import pandas as pd
import torch

def table(path):
    data = torch.load(path)
    rec_key = 'recommentations' if 'recommentations' in data[0].keys() else 'recommendations'
    columns = ['title', 'n_cocitation_score', 'n_context_relevancy', 'n_keyword_relevancy']
    df = pd.DataFrame(data)
    df = df[columns]
    score_columns = df.columns[df.dtypes != 'O']
    total_scores = df[score_columns].sum(axis=0).values.reshape(1, -1)

    total_scores = pd.DataFrame(total_scores, columns=score_columns.to_numpy())
    total_scores.index = ['Total']
        

    st.dataframe(
        df,
        hide_index=True,
        height=562,
    )
    st.dataframe(
        total_scores,
    )


def create(page):
    if page =='eval':
        st.title('Evaluation')
        st.markdown('---')
        st.markdown('Evaluation is crucial for determining which methods yield the best results. It is also essential to develop an evaluation function that aligns with the core objectives of the recommendation system')
        st.markdown('The Evaluation metrics were developed by referencing papers from Microsoft Academic and suggestions from Stanford University students. The proposed evaluation function incorporates cocitation, relevancy, and novelty. Given that novelty is typically determined by whether the author is known or unknown, I opted to use context relevance instead. This adjustment aims to achieve more accurate results based on the contextual information')
        st.header('Dataset')
        st.markdown('---')
        st.markdown('The dataset is constructed by providing the abstracts of 16 influential papers to an LLM model to generate queries. Based on these queries, the recommendation system suggests 30 papers for each influential paper.')

        st.markdown('**Cocitation:** Calculated as the Cocitation between the parent paper and the recommended papers for each influential paper. This metric indicates the frequency with which both papers are referenced together.')
        st.markdown('**Context Relevancy:** Derived from the cosine similarity of the LLM embeddings, providing information on how relevant the recommended papers are to the context of the parent query.')
        st.markdown('**Keyword Relevancy:** An essential metric that assesses the presence of important keywords in both the query and the recommended papers. TF-IDF transformation is applied to the query and recommended papers after preprocessing to determine the keyword relevancy.')
        
        st.header('Matrix Calculation')
        st.markdown('---')
        st.markdown('Considering cocitation, relevancy, and keyword relevancy results in a 16x30 sized metric matrix. The performance of the recommendation system is evaluated by normalizing these 30 metrics for each paper, then averaging the results to obtain a 16x3 matrix (representing cocitation, relevancy, and keyword relevancy for each influential paper). These three metrics are then assigned weights based on their importance to create a comprehensive evaluation function.')
        # st.rerun()
    elif page == 'arch':
        st.title('Research Paper Recommendation System Architecture')
        st.markdown('---')
        st.image('static_files/base_recommentation.png')
        text = '''
                ## Human-LLM Interaction
                ---

                The architecture begins with the model engaging the user in a conversation to gather details about the type of research paper they are looking for. Through a series of questions, the model aims to understand the user's context and preferences. Once sufficient information is collected, the LLM (Large Language Model) generates a query that encapsulates all the necessary details to proceed with the recommendation process.

                ## Query Embedding and Similarity Search
                ---

                1. **Query Embedding:**
                - The user-generated query is transformed into a vector representation using Google embeddings (specifically `models/embedding-001`).
                - This conversion allows the system to process and compare the query effectively in the vector space.

                2. **Similarity Search:**
                - The vectorized query is passed to a vector store where all the research papers are stored in vector format.
                - A cosine similarity search is conducted to identify the top 3 research papers that closely match the query embedding. This ensures that the recommendations are highly relevant to the user's needs.

                ## RAG Chat
                ---

                1. **Recommendation Conversion:**
                - The selected research papers are converted into vectors to facilitate further interactions.
                - A Retrieval-Augmented Generation (RAG) model is created, allowing users to engage in a detailed conversation with the chatbot.

                2. **Interactive QA Chatbot:**
                - Users can ask questions and have a dynamic conversation with the RAG chatbot.
                - This interaction helps users understand the recommended research papers in-depth and gain insights from the content.

                3. **Conclusion:**
                - The system concludes the interaction by summarizing the user's inquiries and providing a comprehensive understanding of the recommended papers.

                ## Flow Summary
                ---

                1. **Human-LLM Interaction:**
                - User answers questions.
                - LLM generates a detailed query.

                2. **Query Embedding and Similarity Search:**
                - Query converted to vectors.
                - Similarity search in vector store.
                - Top 3 papers identified.

                3. **RAG Chat:**
                - Selected papers converted to vectors.
                - RAG model facilitates QA interaction.
                - Users gain detailed insights and conclusions.

                This architecture ensures a personalized and interactive approach to recommending research papers, leveraging advanced embedding and retrieval techniques to enhance user experience and satisfaction.'''
        st.markdown(text)
    elif page == 'res':
        st.header('Google Embeddings')
        st.markdown('---')
        embed_text = '''
            ### Google Embedding for Research Paper Recommendations

            Google embedding is a technique to convert data into vectors. For this project, I used the abstracts of research papers and converted them into chunks of size 800 with a chunk overlap of 40. These numbers were chosen based on intuition and could be experimented with further. My initial thought was to capture as much contextual information as possible for searching.

            In terms of performance, the Google embeddings search system demonstrates high cocitation scores, as well as strong context and keyword relevancy. From my point of view, this method can be considered one of the best options for searching, as it provides good cocitation scores and highly relevant results.
    '''
        st.markdown(embed_text)
        table('norm_scores/google_embed_scores.pt')
        embed_res_text = '''

            As shown in the evaluation section, I took the average of 30 papers to get the cocitation score for each paper. The similarity search using Google embeddings demonstrates a substantial total cocitation score, with strong context and keyword relevancy. These results indicate that Google embeddings offer a robust method for achieving high cocitation scores while maintaining relevance in both context and keywords.
            '''
        st.markdown(embed_res_text)
        st.header('Hybrid Search')
        st.markdown('---')
        hybrid_text = '''
            ### Hybrid Search Approach for Research Paper Recommendations

            Hybrid search combines both dense and sparse representations of vectors, making it a promising approach for search algorithms. In this implementation, I used Google embeddings for the dense representation and BM25 from Pinecone for the sparse representation. These were stored in the Pinecone vector store with dot product as the metric. Since Pinecone has an internal implementation for hybrid search, manual implementation was not necessary. This approach provided results that were comparably similar to context-based searching, although with slightly worse cocitation scores.
            '''
        st.markdown(hybrid_text)
        table('norm_scores/hybrid_scores.pt')
        hybrid_res_text = '''

            The hybrid search method shows slightly lower similarity scores compared to the embedding vectors, possibly because I performed the search without chunking. However, overall, it performs well and provides results that are comparably similar to those obtained through context-based searching.
            '''
        st.markdown(hybrid_res_text)

        st.header('Tf_idf Search')
        st.markdown('---')
        tf_idf_text = '''
            ### Evaluation of TF-IDF for Research Paper Recommendations

            I created the TF-IDF representation with a `max_features` of 15,000 due to the sparsity of the vector and the original dimension being 79,000. To eliminate unnecessary fields, I set `max_df` to 0.85. The corpus was preprocessed by removing unnecessary words such as stop words and punctuation, and by performing lemmatization. After preprocessing, the data was fed into the TF-IDF vectorizer, and the most similar vectors were identified using cosine similarity.
            '''
        st.markdown(tf_idf_text)
        table('norm_scores/tf_idf_scores.pt')
        tf_idf_res_text = '''
            Even though the context relevancy is lower compared to hybrid search and context search, TF-IDF provides high keyword relevancy and pretty good cocitation scores. This approach can still be improved by experimenting with different hyperparameters.'''
        st.header('Observations and Improvements')
        st.markdown('---')
        st.markdown(tf_idf_res_text)
        observations = '''

            1. **Google Embeddings**:
                - Works well with stop words included; removing them might cause issues.
                - Provides comparatively good results.

            2. **Hybrid Search**:
                - Gives good results even without data chunking.
                - Takes in the importance of both sparse and dense representation when searching

            3. **Dataset**:
                - A lack of research papers in specific fields might impact recommendations.
                - The cocitation score API focuses mainly on the hep-th category, causing scores to be zero for papers outside this category.
                - The lack of hep-th papers in the dataset might reduce cocitation values.
                
            4. **TF-IDF**:
                - Sparse representation, limited by max features of 15,000 due to computational constraints.
                - Sometimes misses rare keywords; performs better with larger queries.
                - Using BM25 and Pinecone’s sparse storage feature might yield better results.

            5. **General**:
                - Improved results expected with more training data.
            '''
        st.markdown(observations)

        st.header('Conclusion')
        st.markdown('---')
        conclusion = '''

            The research paper recommendation system developed in this project demonstrates the potential of leveraging various advanced techniques to provide relevant and effective suggestions to users. By integrating methods such as Context based search, Hybrid search, and TF-IDF, the system achieves a comprehensive approach to recommendation.

            In General, the system has demonstrated strong performance across various metrics. The integration of multiple methods ensures that it can cater to diverse user needs. Nevertheless, there is room for improvement. Increasing the dataset size, particularly with more papers from underrepresented categories, and optimizing hyperparameters for TF-IDF could enhance the system's effectiveness further. Additionally, exploring other hybrid search techniques and refining preprocessing steps could lead to even better results.

            Overall, this research paper recommendation system showcases a balanced blend of state-of-the-art techniques and practical considerations, paving the way for more refined and effective academic search solutions in the future.
            '''
        st.markdown(conclusion)

        



if __name__ == '__main__':
    create()