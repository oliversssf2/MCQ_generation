from nltk import sent_tokenize
import torch

def _prepare_inputs_for_ans_extraction(text):
    sents = sent_tokenize(text)
    # print(sents)

    inputs = []
    for i in range(len(sents)):
        source_text = "extract answers:"
        for j, sent in enumerate(sents):
            if i == j:
                sent = "<hl> %s <hl>" % sent
            source_text = "%s %s" % (source_text, sent)
            source_text = source_text.strip()
        
        source_text = source_text + " </s>"
        inputs.append(source_text)

    return sents, inputs

def _prepare_inputs_for_qg_from_answers_hl(sentences, answers):
    inputs = []
    for i, answer in enumerate(answers):
        sents = sentences.copy()
        sent = sents[i]

        answer = answer.strip()
        # print(answer)

        answer_start_index = sent.index(answer)
        sent = '{} <hl> {} <hl> {}'.format(
            sents[i][:answer_start_index], 
            answer, 
            sents[i][(answer_start_index+len(answer)):]
        )
        sents[i] = sent

        source_text = "".join(sents)

        inputs.append(source_text)
    return inputs

def _prepare_inputs_for_e2e_qg_from_answers_hl(contexts):
    inputs = []
    for context in contexts:
        inputs.append("generate questions: " + context)
    return inputs

def get_sent_answer_list(text, ans_ext_model, tokenizer):
    sents, inputs = _prepare_inputs_for_ans_extraction(text)
    encodings = tokenizer.batch_encode_plus(inputs, truncation=True)
    output_ids = ans_ext_model.generate(
        torch.tensor(encodings['input_ids']),
        attention_mask = torch.tensor(encodings['attention_mask']),
        max_length = 32
    )
    answers = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )
    answers = [ ans.strip('<sep>') for ans in answers]
    return sents, answers
