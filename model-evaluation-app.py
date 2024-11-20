import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from korouge_score import rouge_scorer
from konlpy.tag import Okt
import io

def calculate_classification_metrics(actual, predicted):
    """분류 모델의 성능 지표를 계산합니다."""
    metrics = {
        'Accuracy': accuracy_score(actual, predicted),
        'Precision': precision_score(actual, predicted, average='weighted', zero_division=0),
        'Recall': recall_score(actual, predicted, average='weighted', zero_division=0),
        'F1 Score': f1_score(actual, predicted, average='weighted', zero_division=0)
    }
    return metrics

def calculate_rouge_scores(reference_texts, generated_texts):
    """ROUGE 점수를 계산합니다."""
    okt = Okt()
    
    # 형태소 분석
    reference_morph = [" ".join(okt.morphs(text)) for text in reference_texts]
    generated_morph = [" ".join(okt.morphs(text)) for text in generated_texts]
    
    # ROUGE 점수 계산
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_results = []
    
    for ref, gen in zip(reference_morph, generated_morph):
        scores = scorer.score(ref, gen)
        rouge_results.append({
            'Reference Summary': ref,
            'Generated Summary': gen,
            'ROUGE-1 Precision': scores['rouge1'].precision,
            'ROUGE-1 Recall': scores['rouge1'].recall,
            'ROUGE-1 F1': scores['rouge1'].fmeasure,
            'ROUGE-2 Precision': scores['rouge2'].precision,
            'ROUGE-2 Recall': scores['rouge2'].recall,
            'ROUGE-2 F1': scores['rouge2'].fmeasure,
            'ROUGE-L Precision': scores['rougeL'].precision,
            'ROUGE-L Recall': scores['rougeL'].recall,
            'ROUGE-L F1': scores['rougeL'].fmeasure
        })
    
    return pd.DataFrame(rouge_results)

def main():
    st.title("LLM Evaluation (Korean)")
    
    # Task 선택
    task = st.radio("평가할 모델 유형을 선택하세요:", ["분류", "생성/요약"])
    
    # 파일 업로드
    uploaded_file = st.file_uploader("테스트 데이터 파일을 업로드하세요 (Excel)", type=['xlsx'])
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("데이터 미리보기:")
        st.dataframe(df.head())
        
        columns = df.columns.tolist()
        
        if task == "분류":
            st.subheader("분류 모델 평가")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("실제값 컬럼 선택")
                actual_cols = st.multiselect("실제값 컬럼을 선택하세요:", columns, key='actual_cls')
            
            with col2:
                st.write("예측값 컬럼 선택")
                predicted_cols = st.multiselect("예측값 컬럼을 선택하세요:", columns, key='pred_cls')
            
            if st.button("성능 평가 실행"):
                if len(actual_cols) != len(predicted_cols):
                    st.error("실제값과 예측값 컬럼 수가 일치해야 합니다.")
                    return
                
                results = []
                for actual_col, predicted_col in zip(actual_cols, predicted_cols):
                    metrics = calculate_classification_metrics(
                        df[actual_col].astype(str), 
                        df[predicted_col].astype(str)
                    )
                    results.append({
                        'Level': f"{actual_col} vs {predicted_col}",
                        **metrics
                    })
                
                results_df = pd.DataFrame(results)
                st.write("평가 결과:")
                st.dataframe(results_df)
                
                # 결과 다운로드
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='성능측정결과', index=False)
                
                fn_result = f"{uploaded_file.name.split('.')[0]}-cls-result.xlsx"
                
                st.download_button(
                    label="평가 결과 다운로드",
                    data=output.getvalue(),
                    file_name=fn_result,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
        else:  # 생성/요약 모델
            st.subheader("생성/요약 모델 평가")
            
            col1, col2 = st.columns(2)
            with col1:
                reference_col = st.selectbox("실제 텍스트 컬럼을 선택하세요:", columns, key='actual_gen')
            
            with col2:
                generated_col = st.selectbox("생성된 텍스트 컬럼을 선택하세요:", columns, key='pred_gen')
            
            if st.button("성능 평가 실행"):
                # 결측치 제거
                mask = (df[reference_col].notna()) & (df[generated_col].notna())
                df_clean = df[mask]
                
                if df_clean.empty:
                    st.error("유효한 데이터가 없습니다.")
                    return
                
                # ROUGE 점수 계산
                rouge_df = calculate_rouge_scores(
                    df_clean[reference_col].astype(str),
                    df_clean[generated_col].astype(str)
                )
                
                st.write("ROUGE 점수 결과:")
                st.dataframe(rouge_df)
                
                # 평균 점수 계산
                avg_scores = rouge_df[[col for col in rouge_df.columns if col.startswith('ROUGE')]].mean()
                st.write("평균 ROUGE 점수:")
                st.dataframe(pd.DataFrame([avg_scores]))
                
                # 결과 다운로드
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    rouge_df.to_excel(writer, sheet_name='문장별_성능측정결과', index=False)
                    pd.DataFrame([avg_scores]).to_excel(writer, sheet_name='평균_성능측정결과')
                    
                fn_result = f"{uploaded_file.name.split('.')[0]}-rouge-result.xlsx"
                
                st.download_button(
                    label="평가 결과 다운로드",
                    data=output.getvalue(),
                    file_name=fn_result,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
