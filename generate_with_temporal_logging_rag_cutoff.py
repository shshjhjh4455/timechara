import os
import json
import openai
import argparse
from tqdm import tqdm
from rich import print as rprint
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    preprocess_generation,
    call_openai_api,
    extract_and_format_number,
    compare_book_chapters,
    series_name_dict,
    character_period,
)

openai.api_key = os.getenv('OPENAI_API_KEY')

def load_final_temporal_data():
    """Load final_temporal_data.json for RAG"""
    with open('data/final_temporal_data.json', 'r') as f:
        return json.load(f)

def find_raw_text_for_predicted_chapter(predicted_book_chapter, final_temporal_data):
    """Find raw_text matching the predicted book-chapter"""
    if not predicted_book_chapter:
        return None

    # Convert format: "7-16" -> "Book7-chapter16"
    if not predicted_book_chapter.startswith('Book'):
        parts = predicted_book_chapter.split('-')
        if len(parts) == 2:
            target_period = f"Book{parts[0]}-chapter{parts[1]}"
        else:
            return None
    else:
        target_period = predicted_book_chapter

    # Find matching scene
    for scene in final_temporal_data:
        if scene.get('question_period') == target_period:
            return scene.get('raw_text', '')

    return None

def generate_response_with_raw_text_cutoff(model_name, character, character_period, question, predicted_book_chapter, final_temporal_data):
    """Generate response using ONLY the raw text from predicted book-chapter (RAG-cutoff)"""

    # Find raw text for predicted chapter
    matching_raw_text = find_raw_text_for_predicted_chapter(predicted_book_chapter, final_temporal_data)

    if not matching_raw_text:
        # No raw text available - generate knowledge-cutoff response
        rag_prompt = f"""You are {character} from a story. You are currently at: {character_period}

Someone asks you: "{question}"

I don't have access to the specific text passage that would help me answer this question accurately. Without the relevant text, I cannot provide detailed information about this particular moment or event.

Your response:"""
    else:
        # Use ONLY the raw text - strict knowledge cutoff
        rag_prompt = f"""You are {character} from a story. You are currently at: {character_period}

Someone asks you: "{question}"

Here is the relevant text passage from this scene:

---
{matching_raw_text[:2000]}...
---

Answer the question based ONLY on the information provided in the text passage above.

STRICT RULES:
- Use ONLY information explicitly stated in the provided text passage
- Do NOT use any external knowledge about stories, characters, or events
- Do NOT reference specific book titles, character full names, or locations unless mentioned in the text
- If the text doesn't contain sufficient information to answer the question, state this honestly
- Respond as {character} would, but only using what is shown in the provided text

Your response:"""

    # Generate response
    completion = call_openai_api(model_name, "You are a helpful assistant that follows instructions precisely.", rag_prompt,
                                 max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
    return completion.choices[0].message.content

def narrative_experts_with_rag_cutoff(model_name, system_prompt, question, question_period, character, book_no, day, participants, series_name, final_temporal_data):
    """Modified narrative_experts with RAG-cutoff for response generation"""
    hint = []
    temporal_predictions = {}

    if series_name == 'harry_potter':
        try:
            character_book_chapter_num_abs = character_period[series_name_dict[series_name]][day.lower()][int(book_no) - 1]
        except:
            character_book_chapter_num_abs = f'{book_no}-0'
        book_chapter_name = f'book number and chapter number'
        book_chapter_format = f'book M - chapter N (write Arabic number instead of Roman number for the volume number)'
        book_chapter_cnt = 2
        series_full_name = 'Harry Potter'
    else:
        raise ValueError("Only Harry Potter supported in this version")

    # Temporal Expert (unchanged)
    user_prompt = f"""You will be given a question from {series_full_name} series at a specific time. Your task is to identify the exact {book_chapter_name} of the scene of the question. Below is the data:
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

    completion = call_openai_api(model_name, "You are a helpful and accurate assistant.", user_prompt,
                                 max_tokens=1024, temperature=0.0, top_p=0.95, n=1, seed=0)
    content = completion.choices[0].message.content

    # Extract temporal prediction
    temporal_label_str = extract_and_format_number(content.strip().split('\n')[-1].lower(), book_chapter_cnt)

    # Store predictions
    temporal_predictions['predicted'] = temporal_label_str
    temporal_predictions['actual'] = question_period
    temporal_predictions['character_position'] = character_book_chapter_num_abs
    temporal_predictions['temporal_expert_response'] = content

    # Compare with character's position
    is_future = False
    if temporal_label_str == '':
        temporal_predictions['temporal_relation'] = 'unknown'
    else:
        temporal_label = compare_book_chapters(temporal_label_str, character_book_chapter_num_abs)
        temporal_predictions['temporal_relation'] = temporal_label
        if temporal_label == 'after':
            hint.append(f"Note that the period of the question is in the future relative to {character}'s time point. Therefore, you should not answer the question or mention any facts that occurred after {character}'s time point.")
            is_future = True
        elif temporal_label == 'before':
            pass
        else:
            raise ValueError

    # Response generation with RAG-cutoff
    if is_future:
        response = f"I cannot answer this question as it refers to events that haven't happened yet from my perspective."
    else:
        # Use RAG-cutoff: only raw text from predicted book-chapter
        response = generate_response_with_raw_text_cutoff(
            model_name, character, day, question, temporal_label_str, final_temporal_data
        )

    return response, '\n'.join(hint), temporal_predictions

def process_single_example_with_rag_cutoff(args, example, data_idx, final_temporal_data):
    """Process a single example with RAG-cutoff"""
    try:
        series = example['series']
        character = example['character']
        character_period = example['character_period']
        question = example['question']
        question_period = example['question_period']
        participants = example['participants']

        system_prompt, book_no, day, day_character = preprocess_generation(series, character, character_period)

        response, hint, temporal_predictions = narrative_experts_with_rag_cutoff(
            args.model_name, system_prompt, question, question_period,
            character, book_no, day, participants, series, final_temporal_data
        )

        output = {
            'data_idx': data_idx,
            'response': response,
            'thought': hint,
            'temporal_predictions': temporal_predictions,
            'method': 'narrative-experts-RAG-cutoff'
        }

        return data_idx, output, None

    except Exception as e:
        return data_idx, None, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='gpt-4o-2024-05-13')
    parser.add_argument("--data_file", default='data/valid_harrypotter_with_raw_text.json')
    parser.add_argument("--output_fname", default='timechara_rag_cutoff_results.json')
    parser.add_argument("--max_workers", default=5, type=int, help="Number of parallel workers")
    args = parser.parse_args()

    # Load dataset
    with open(args.data_file, 'r') as f:
        dataset = json.load(f)

    # Load final_temporal_data for RAG
    final_temporal_data = load_final_temporal_data()
    print(f"Loaded {len(final_temporal_data)} scenes from final_temporal_data.json")

    print(f"Processing {len(dataset)} examples with RAG-cutoff method...")

    # Process examples in parallel
    outputs = [None] * len(dataset)
    errors = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_example_with_rag_cutoff, args, example, data_idx, final_temporal_data): data_idx
            for data_idx, example in enumerate(dataset)
        }

        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc="Processing RAG-cutoff", ncols=120):
            data_idx = future_to_idx[future]
            try:
                result_idx, output, error = future.result()
                if error:
                    errors.append(f"Index {result_idx}: {error}")
                    print(f"Error at index {result_idx}: {error}")
                else:
                    outputs[result_idx] = output
            except Exception as exc:
                errors.append(f"Index {data_idx}: {exc}")
                print(f"Exception at index {data_idx}: {exc}")

    # Filter out None values
    valid_outputs = [output for output in outputs if output is not None]

    print(f"\nCompleted: {len(valid_outputs)}/{len(dataset)} examples")
    print(f"Using RAG-cutoff method: responses generated using ONLY raw text from predicted book-chapters")

    # Save outputs
    output_path = f'outputs/{args.output_fname}'
    os.makedirs('outputs', exist_ok=True)
    with open(output_path, 'w') as fp:
        json.dump(valid_outputs, fp, indent=4)
    print(f"Saved RAG-cutoff outputs to {output_path}")

if __name__ == '__main__':
    main()