import os
import copy
import json
import openai
import argparse
from tqdm import tqdm
from rich import print as rprint
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    character_dict,
    preprocess_evaluation,
    call_openai_api,
    extract_score,
)

openai.api_key = os.getenv('OPENAI_API_KEY')

def evaluate_spatiotemporal_consistency(model_name, spatiotemporal_prompt_template, agent_name, question, answer, agent_fact):
    prompt = spatiotemporal_prompt_template.format(agent_name=agent_name,
                                                   question_0=question,
                                                   answer_0=answer,
                                                   agent_fact_0=agent_fact)
    completion = call_openai_api(model_name, 'You are a helpful and accurate assistant.', prompt)
    content = completion.choices[0].message.content
    return content

def evaluate_personality_consistency(model_name, personality_prompt_template, agent_name, question, answer, agent_personality):
    prompt = personality_prompt_template.format(agent_name=agent_name,
                                                question_0=question,
                                                response_0=answer,
                                                agent_personality=agent_personality)
    completion = call_openai_api(model_name, 'You are a helpful and accurate assistant.', prompt)
    content = completion.choices[0].message.content
    return content

def process_single_evaluation(args, dataset_example, response_example, data_idx, personality, spatiotemporal_prompt_template, personality_prompt_template):
    """Process a single evaluation example"""
    try:
        series = dataset_example['series']
        question = dataset_example['question']
        response = response_example['response']
        hint = response_example['thought']
        character = dataset_example['character']
        character_period = dataset_example['character_period']

        output = {
            'data_idx': data_idx,
            'response': response,
            'thought': hint,
            'temporal_eval': '',
            'temporal_score': '',
            'spatial_eval': '',
            'spatial_score': '',
            'personality_eval': '',
            'personality_score': '',
        }

        day_character = preprocess_evaluation(series, character, character_period)

        # spatiotemporal consistency
        if args.eval_mode in ['all', 'spatiotemporal']:
            data_type = dataset_example['data_type']
            temporal_label = dataset_example['temporal_label']
            spatial_label = dataset_example['spatial_label']

            if 'future' in data_type:
                content_temporal = evaluate_spatiotemporal_consistency(
                    args.eval_model_name, spatiotemporal_prompt_template, day_character, question, response, temporal_label)
                temporal_consistency_score, _ = extract_score(content_temporal)
                output['temporal_eval'] = copy.deepcopy(content_temporal)
                output['temporal_score'] = copy.deepcopy(temporal_consistency_score)
            elif 'past' in data_type:
                if data_type in ['past-absence', 'past-presence']:
                    content_spatial = evaluate_spatiotemporal_consistency(
                        args.eval_model_name, spatiotemporal_prompt_template, day_character, question, response, spatial_label)
                    spatial_consistency_score, _ = extract_score(content_spatial)
                    # check spatial consistency
                    if spatial_consistency_score == '0':
                        content_temporal = 'Spatiotemporally inconsistent'
                        temporal_consistency_score = '0'
                    else:
                        content_temporal = evaluate_spatiotemporal_consistency(
                            args.eval_model_name, spatiotemporal_prompt_template, day_character, question, response, temporal_label)
                        temporal_consistency_score, _ = extract_score(content_temporal)
                    output['spatial_eval'] = copy.deepcopy(content_spatial)
                    output['spatial_score'] = copy.deepcopy(spatial_consistency_score)
                    output['temporal_eval'] = copy.deepcopy(content_temporal)
                    output['temporal_score'] = copy.deepcopy(temporal_consistency_score)
                elif data_type == 'past-only':
                    content_temporal = evaluate_spatiotemporal_consistency(
                        args.eval_model_name, spatiotemporal_prompt_template, day_character, question, response, temporal_label)
                    temporal_consistency_score, _ = extract_score(content_temporal)
                    output['temporal_eval'] = copy.deepcopy(content_temporal)
                    output['temporal_score'] = copy.deepcopy(temporal_consistency_score)

        # personality consistency
        if args.eval_mode in ['all', 'personality']:
            content_personality = evaluate_personality_consistency(
                args.eval_model_name, personality_prompt_template, day_character, question, response, personality[character])
            personality_consistency_score, _ = extract_score(content_personality)
            output['personality_eval'] = copy.deepcopy(content_personality)
            output['personality_score'] = copy.deepcopy(personality_consistency_score)

        return data_idx, output, None

    except Exception as e:
        return data_idx, None, str(e)

def print_spatiotemporal_scores(dataset, outputs):
    spatiotemporal_score = {'future': [], 'past-absence': [], 'past-presence': [], 'past-only': []}
    for data_idx, example in enumerate(dataset):
        if data_idx >= len(outputs):
            break
        output = outputs[data_idx]
        data_type = example['data_type']
        if output['temporal_score'] != '' and output['temporal_score'] != '.':
            try:
                if data_type in ['future', 'past-only']:
                    spatiotemporal_score[data_type].append(float(output['temporal_score']))
                elif data_type in ['past-absence', 'past-presence']:
                    if output['spatial_score'] != '' and output['spatial_score'] != '.':
                        spatiotemporal_score[data_type].append(float(output['temporal_score']) * float(output['spatial_score']))
            except ValueError:
                continue

    combined_scores = []
    for value in spatiotemporal_score.values():
        combined_scores.extend(value)

    print(f'\n*** Spatiotemporal Scores ***\n')
    for k,v in spatiotemporal_score.items():
        if v:
            print(f'{k} (max 1.0): {sum(v)}/{len(v)} (={round(sum(v)/len(v),5)})')
    print(f'\nTotal (max 1.0): {sum(combined_scores)}/{len(combined_scores)} (={round(sum(combined_scores)/len(combined_scores),5)})')

def print_personality_scores(dataset, outputs):
    combined_scores = []
    for data_idx, example in enumerate(dataset):
        if data_idx >= len(outputs):
            break
        output = outputs[data_idx]
        if output['personality_score'] != '' and output['personality_score'] != '.':
            try:
                combined_scores.append(float(output['personality_score']))
            except ValueError:
                continue

    print(f'\n*** Personality Scores ***\n')
    if combined_scores:
        print(f'\nTotal (max 7.0): {sum(combined_scores)}/{len(combined_scores)} (={round(sum(combined_scores)/len(combined_scores),5)})')
    else:
        print('\nNo valid personality scores found')

def main():
    assert openai.api_key is not None, f"Export your OPENAI_API_KEY!"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='gpt-4-1106-preview', choices=['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'llama-2-13b-chat', 'mistral-instruct-7b',])
    parser.add_argument("--eval_model_name", default='gpt-4-1106-preview', choices=['gpt-4o-2024-05-13', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106'])
    parser.add_argument("--method_name", default='zero-shot', choices=['zero-shot','zero-shot-cot', 'few-shot', 'self-refine', 'rag-cutoff', 'narrative-experts', 'narrative-experts-rag-cutoff'])
    parser.add_argument("--data_file", default='data/timechara_test_600.json', type=str, help="input data file")
    parser.add_argument("--eval_mode", default='spatiotemporal', choices=['spatiotemporal', 'personality', 'all'])
    parser.add_argument("--output_dir", default='outputs/', type=str, help="output directoy")
    parser.add_argument("--input_fname", default='generated.json', type=str, help="input file name ")
    parser.add_argument("--output_fname", default='evaluated.json', type=str, help="output file name")
    parser.add_argument("--max_samples", default=None, type=int, help="maximum number of samples to evaluate")
    parser.add_argument("--max_workers", default=30, type=int, help="maximum number of parallel workers for evaluation")
    args = parser.parse_args()

    # Load local dataset
    with open(args.data_file, 'r') as f:
        dataset = json.load(f)

    # Load generated responses
    with open(os.path.join(args.output_dir, 'test', f'{args.model_name}_{args.method_name}', args.input_fname), 'r') as fp:
        responses = json.load(fp)

    # Apply max_samples if specified
    if args.max_samples:
        dataset = dataset[:args.max_samples]
        responses = responses[:args.max_samples]

    assert len(dataset) == len(responses), f"Error: # dataset != # generated responses"

    personality = dict.fromkeys(character_dict, '')
    for k,v in character_dict.items():
        with open(f'data/personality/{v}_personality.txt', 'r', encoding='utf-8') as fp:
            personality[k] = fp.read()

    with open('data/spatiotemporal_consistency_evaluation_prompt.txt', 'r') as fp:
        spatiotemporal_prompt_template = fp.read()
    with open('data/personality_consistency_evaluation_prompt.txt', 'r') as fp:
        personality_prompt_template = fp.read()

    print(f"Processing {len(dataset)} evaluations with {args.max_workers} parallel workers...")

    # Process evaluations in parallel
    outputs = [None] * len(dataset)
    errors = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                process_single_evaluation,
                args,
                dataset[data_idx],
                responses[data_idx],
                data_idx,
                personality,
                spatiotemporal_prompt_template,
                personality_prompt_template
            ): data_idx
            for data_idx in range(len(dataset))
        }

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc="Evaluating", ncols=120):
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

    print(f"\nCompleted: {len(valid_outputs)}/{len(dataset)} evaluations")
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  {error}")

    # Save outputs
    os.makedirs(os.path.join(args.output_dir, 'test', f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}'), exist_ok=True)
    with open(os.path.join(args.output_dir, 'test', f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}', args.output_fname), 'w') as fp:
        json.dump(valid_outputs, fp, indent=4)
    print(f"\n\nSaved outputs to {os.path.join(args.output_dir, 'test', f'{args.model_name}_{args.method_name}', f'eval_{args.eval_mode}', args.output_fname)}!!\n\n")

    # print scores
    if args.eval_mode in ['spatiotemporal', 'all']:
        print_spatiotemporal_scores(dataset, valid_outputs)
    if args.eval_mode in ['personality', 'all']:
        print_personality_scores(dataset, valid_outputs)

    print('Good Job Computer!')

if __name__ == '__main__':
    main()