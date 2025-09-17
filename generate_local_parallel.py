import os
import json
import openai
import argparse
from tqdm import tqdm
from rich import print as rprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from utils import (
    preprocess_generation,
    call_openai_api,
    call_opensource_model,
)
from methods.self_refine.self_refine import self_refine
from methods.rag.rag import RAGCutoff
from methods.narrative_experts.narrative_experts import (
    narrative_experts,
    NarrativeExpertsRAGCutoff,
)

openai.api_key = os.getenv('OPENAI_API_KEY')

def process_single_example(args, example, data_idx, few_shot_dict=None, rag=None):
    """Process a single example"""
    try:
        series = example['series']
        data_type = example['data_type']
        character = example['character']
        character_period = example['character_period']
        question = example['question']
        question_period = example['question_period']
        participants = example['participants']

        system_prompt, book_no, day, day_character = preprocess_generation(series, character, character_period)

        if args.model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
            if args.method_name == 'zero-shot':
                completion = call_openai_api(args.model_name, system_prompt, question,
                                             max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
                response = completion.choices[0].message.content
                hint = '-'
            elif args.method_name == 'zero-shot-cot':
                question_zero_shot_cot = f"{question}\nLet's think step by step."
                completion = call_openai_api(args.model_name, system_prompt, question_zero_shot_cot,
                                             max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
                response = completion.choices[0].message.content
                hint = '-'
            elif args.method_name == 'few-shot':
                assert few_shot_dict is not None
                few_shot_examples = few_shot_dict[f'{character}_{character_period}']
                question_few_shot = f"{few_shot_examples}\n{question}"
                completion = call_openai_api(args.model_name, system_prompt, question_few_shot,
                                             max_tokens=2048, temperature=0.2, top_p=1.0, n=1, seed=0)
                response = completion.choices[0].message.content
                hint = '-'
            elif args.method_name == 'self-refine':
                response, hint = self_refine(args.model_name, system_prompt, question)
            elif args.method_name == 'rag-cutoff':
                assert rag is not None
                response, hint = rag.generate(args.model_name, system_prompt, question, character, series, book_no, day)
            elif args.method_name == 'narrative-experts':
                response, hint, _ = narrative_experts(args.model_name, system_prompt, question, question_period, character, book_no, day, participants, series)
            elif args.method_name == 'narrative-experts-rag-cutoff':
                assert rag is not None
                response, hint, _ = rag.generate(args.model_name, system_prompt, question, question_period, character, book_no, day, participants, series)
            else:
                raise NotImplementedError

        output = {
            'data_idx': data_idx,
            'response': response,
            'thought': hint,
        }

        return data_idx, output, None

    except Exception as e:
        return data_idx, None, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='gpt-4-1106-preview', choices=['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'llama-2-13b-chat', 'mistral-instruct-7b'])
    parser.add_argument("--method_name", default='zero-shot', choices=['zero-shot','zero-shot-cot', 'few-shot', 'self-refine', 'rag-cutoff', 'narrative-experts', 'narrative-experts-rag-cutoff'])
    parser.add_argument("--data_file", default='data/timechara_test_600.json', type=str, help="input data file")
    parser.add_argument("--output_dir", default='outputs/', type=str, help="output directoy")
    parser.add_argument("--output_fname", default='generated.json', type=str, help="output file name")
    parser.add_argument("--rag_cache_dir", default='methods/rag/text-embedding-ada-002/cutoff', type=str, help="RAG cache directory")
    parser.add_argument("--max_samples", default=None, type=int, help="maximum number of samples to process")
    parser.add_argument("--max_workers", default=50, type=int, help="maximum number of parallel workers (Tier 5: up to 100)")
    args = parser.parse_args()

    # Load local dataset
    with open(args.data_file, 'r') as f:
        dataset = json.load(f)

    if args.max_samples:
        dataset = dataset[:args.max_samples]

    # Load Model requirements
    if args.model_name in ['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106']:
        assert openai.api_key is not None, f"Export your OPENAI_API_KEY!"
    else:
        raise NotImplementedError("Parallel processing only supports OpenAI models")

    # Load requirements
    few_shot_dict = None
    rag = None

    if args.method_name == 'few-shot':
        few_shot_dict = dict()
        for series_name in ['harry_potter', 'the_lord_of_the_rings', 'twilight', 'hunger_games']:
            with open(f'methods/few_shot/{series_name}.json', 'r') as fp:
                current_dict = json.load(fp)
            few_shot_dict.update(current_dict)
    elif args.method_name == 'rag-cutoff':
        rag = RAGCutoff(openai.api_key, args.rag_cache_dir)
    elif args.method_name == 'narrative-experts-rag-cutoff':
        rag = NarrativeExpertsRAGCutoff(openai.api_key, args.rag_cache_dir)

    print(f"Processing {len(dataset)} examples with {args.max_workers} parallel workers...")

    # Process examples in parallel
    outputs = [None] * len(dataset)
    errors = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single_example, args, example, data_idx, few_shot_dict, rag): data_idx
            for data_idx, example in enumerate(dataset)
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc="Processing", ncols=120):
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

    # Filter out None values (failed requests)
    valid_outputs = [output for output in outputs if output is not None]

    print(f"\nCompleted: {len(valid_outputs)}/{len(dataset)} examples")
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")

    # Save outputs
    os.makedirs(os.path.join(args.output_dir, 'test', f'{args.model_name}_{args.method_name}'), exist_ok=True)
    with open(os.path.join(args.output_dir, 'test', f'{args.model_name}_{args.method_name}', args.output_fname), 'w') as fp:
        json.dump(valid_outputs, fp, indent=4)
    print(f"\n\nSaved outputs to {os.path.join(args.output_dir, 'test', f'{args.model_name}_{args.method_name}', args.output_fname)}!!\n\n")

    print('Good Job Computer!')

if __name__ == '__main__':
    main()