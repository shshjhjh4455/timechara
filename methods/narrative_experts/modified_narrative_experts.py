"""
Modified Narrative Experts - No Before/After Conversion, No Metadata
목적: Temporal Expert의 실제 예측 정확도만 측정
"""

from rich import print as rprint

from utils import (
    series_name_dict,
    character_period,
    call_openai_api,
    extract_and_format_number,
)
from methods.rag.rag import RAGCutoff


class ModifiedNarrativeExpertsRAG(RAGCutoff):
    """
    Modified Narrative Experts with RAG

    핵심 변경사항:
    1. ❌ Before/After 변환 로직 제거 (우연히 맞추는 케이스 제거)
    2. ✅ Temporal Expert 예측 결과를 그대로 사용
    3. ✅ 예측된 Book-Chapter의 raw text만 사용
    4. ❌ 프롬프트에서 메타정보 제거 (책 이름, 캐릭터 이름, Book-Chapter)
    """

    def __init__(self, openai_api_key, rag_cache_dir):
        super().__init__(openai_api_key, rag_cache_dir)

    def get_chapter_raw_text(self, series_name, predicted_chapter):
        """
        예측된 Book-Chapter의 raw text만 추출 (메타정보 없이)

        Args:
            series_name: 'harry_potter' 등
            predicted_chapter: '5-8' 형식

        Returns:
            raw_text: 메타정보가 제거된 순수 텍스트
        """
        # RAG에서 해당 챕터의 문서 검색
        # 임시 쿼리로 해당 챕터 문서 검색
        all_docs = self.vectorstore[series_name].get()

        # predicted_chapter에 해당하는 문서만 필터링
        chapter_docs = []
        for i, doc_metadata in enumerate(all_docs['metadatas']):
            source = doc_metadata.get('source', '')
            doc_chapter = extract_and_format_number(source, 2)  # Harry Potter는 2 (book-chapter)

            if doc_chapter == predicted_chapter:
                chapter_docs.append(all_docs['documents'][i])

        # 모든 chunk를 합쳐서 반환
        if chapter_docs:
            return '\n\n'.join(chapter_docs)
        else:
            return ""

    def generate(self,
                 model_name,
                 system_prompt,
                 question,
                 question_period,
                 character,
                 book_no,
                 day,
                 participants,
                 series_name,
                 data_type=None,
                 max_tokens=1024,
                 temperature=0.0,
                 top_p=0.95,
                 n=1,
                 seed=0):
        """
        Modified Narrative Experts 생성 로직

        순서:
        1. Temporal Expert: Book-Chapter 예측
        2. Spatial Expert: Present/Absent 판단
        3. ❌ Before/After 변환 없음
        4. 예측된 챕터의 raw text 추출 (메타정보 제거)
        5. 답변 생성
        """
        hint = []

        # Series-specific 설정
        if series_name == 'harry_potter':
            try:
                character_book_chapter_num_abs = character_period[series_name_dict[series_name]][day.lower()][int(book_no) - 1]
            except:
                character_book_chapter_num_abs = f'{book_no}-0'
            book_chapter_name = f'book number and chapter number'
            book_chapter_format = f'book M - chapter N'
            book_chapter_cnt = 2
        else:
            raise NotImplementedError("Only Harry Potter is supported in this modified version")

        series_full_name = series_name_dict[series_name]

        # ========================================
        # Step 1: Temporal Expert (Book-Chapter 예측)
        # ========================================
        temporal_prompt = f"""You will be given a question from {series_full_name} series. Your task is to identify the exact {book_chapter_name} of the scene of the question. Below is the data:
***
[Question]
{question}
***
[Identification Criterion]
What is the exact {book_chapter_name} of the scene of the question?

[Identification Steps]
1. Read through the [Question], recall the scene from the question, and describe it using the six Ws (Who, What, When, Where, Why, and How).
2. Identify the exact {book_chapter_name} of the scene of the question, in '{book_chapter_format}' format.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, print the output on its own line corresponding to the correct answer. At the end, repeat just the selected output again by itself on a new line."""

        if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4o-mini', 'gpt-4o-nano']:
            completion = call_openai_api(model_name, "You are a helpful and accurate assistant.", temporal_prompt,
                                        max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
            temporal_content = completion.choices[0].message.content
            print(f"[Temporal Expert] # of prompt tokens = {completion.usage.prompt_tokens}")
            print(f"[Temporal Expert] # of completion tokens = {completion.usage.completion_tokens}")
        else:
            raise NotImplementedError("Only OpenAI models supported")

        rprint(f"[purple]Temporal Expert Prediction:[/purple]\n{temporal_content[:200]}...")

        # Extract predicted book-chapter
        temporal_label_str = extract_and_format_number(temporal_content.strip().split('\n')[-1].lower(), book_chapter_cnt)
        rprint(f"[bold purple]Predicted: {temporal_label_str}, Gold: {question_period}[/bold purple]")

        # ❌ Before/After 변환 로직 제거 - 예측을 그대로 사용

        # ========================================
        # Step 2: Spatial Expert (Present/Absent 판단)
        # ========================================
        spatial_prompt = f"""You will be given a question and a character from {series_full_name} series. Your task is to classify whether the character is a participant (i.e., present or absent) in the scene of the question. Below is the data:
***
[Question]
{question}
[Character]
{character}
***
[Classification Criterion]
Is the character a participant in the scene of the question?

[Classification Steps]
1. Read through the [Question], recall the scene from the question, and describe it using the six Ws (Who, What, When, Where, Why, and How).
2. Identify the exact {book_chapter_name} of the scene of the question.
3. Write a list of every character involved in the scene described in the question, including those not explicitly mentioned in the question but who were present in the scene.
4. Compare the list of participants to the character. Check if the list of participants contains the character.
5. If the list contains the character, classify it as 'present'. Otherwise, classify it as 'absent'.

First, write out in a step by step manner your reasoning about the criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then, print the output on its own line corresponding to the correct answer. At the end, repeat just the selected output again by itself on a new line."""

        if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4o-mini', 'gpt-4o-nano']:
            completion = call_openai_api(model_name, "You are a helpful and accurate assistant.", spatial_prompt,
                                        max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
            spatial_content = completion.choices[0].message.content
            print(f"[Spatial Expert] # of prompt tokens = {completion.usage.prompt_tokens}")
            print(f"[Spatial Expert] # of completion tokens = {completion.usage.completion_tokens}")
        else:
            raise NotImplementedError("Only OpenAI models supported")

        rprint(f"[cyan]Spatial Expert Prediction:[/cyan]\n{spatial_content[:200]}...")

        # Extract spatial label
        spatial_label_str = spatial_content.strip().split('\n')[-1].lower()
        if 'absent' in spatial_label_str and 'present' not in spatial_label_str:
            # ❌ 캐릭터 이름도 제거 - "the character"로 대체
            hint.append(f"Note that the character had not participated in the scene described in the question.")
            rprint(f"[bold cyan]Spatial: ABSENT[/bold cyan]")
        else:
            rprint(f"[bold cyan]Spatial: PRESENT[/bold cyan]")

        # ========================================
        # Step 3: RAG - 예측된 챕터의 raw text 추출
        # ========================================
        raw_context = ""
        if temporal_label_str:
            raw_context = self.get_chapter_raw_text(series_name, temporal_label_str)
            rprint(f"[yellow]RAG Context Length: {len(raw_context)} characters[/yellow]")

        # ========================================
        # Step 4: 답변 생성 (메타정보 완전히 제거)
        # ========================================
        # ❌ System prompt에서 책 이름, 캐릭터 이름, Book-Chapter, Period 정보 제거
        # ✅ Raw text만 context로 제공
        # ✅ Generic role-playing prompt만 사용

        # 메타데이터 없는 System Prompt
        generic_system_prompt = "You are roleplaying as a character. Answer the question based only on the provided context. If the context is not sufficient to answer the question, say you don't know or weren't present."

        if raw_context:
            # Context 있으면 제공 (메타정보 제거)
            context_prompt = f"Context: {raw_context[:2000]}\n***\n"  # 최대 2000자로 제한
        else:
            context_prompt = ""

        # Final query with optional hint
        if len(hint) > 0:
            final_query = f"{context_prompt}{question}\n(HINT: {' '.join(hint)})"
        else:
            final_query = f"{context_prompt}{question}"

        # Generate final response (메타데이터 없는 system prompt 사용)
        if model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'gpt-4o-mini', 'gpt-4o-nano']:
            completion = call_openai_api(model_name, generic_system_prompt, final_query,
                                        max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, seed=seed)
            response = completion.choices[0].message.content
            print(f"[Final Response] # of prompt tokens = {completion.usage.prompt_tokens}")
            print(f"[Final Response] # of completion tokens = {completion.usage.completion_tokens}")
        else:
            raise NotImplementedError("Only OpenAI models supported")

        rprint(f"[green]Final Response:[/green]\n{response[:200]}...")

        return response, '\n'.join(hint), temporal_label_str
