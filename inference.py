# import os
# import json
# from PIL import Image
# import torch
# import torchvision.transforms as T
# from models import build_model
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import argparse

# # ==== ARGPARSE ====
# def get_args():
#     parser = argparse.ArgumentParser(description="RelTR inference + translate scene graph")
#     parser.add_argument('--img_dir', type=str, required=True, help="Folder of test images")
#     parser.add_argument('--output_dir', type=str, default="outputs_reltr", help="Folder to save JSON results")
#     parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
#     parser.add_argument('--checkpoint', type=str, required=True, help="Path to RelTR checkpoint")
#     parser.add_argument('--translation_model', type=str, default="Qwen/Qwen3-8B", help="HuggingFace model for translation")
#     parser.add_argument('--threshold', type=float, default=0.35, help="Confidence threshold for keeping relations")
#     parser.add_argument('--topk', type=int, default=10, help="Top-K relations per image")
#     return parser.parse_args()

# # ==== LABELS ====
# CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
#             'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
#             'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
#             'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
#             'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
#             'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
#             'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
#             'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
#             'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
#             'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
#             'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
#             'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
#             'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
#             'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

# REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
#                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
#                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
#                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
#                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
#                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

# # ==== UTILS ====
# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=1)

# def rescale_bboxes(out_bbox, size, device):
#     img_w, img_h = size
#     b = box_cxcywh_to_xyxy(out_bbox)
#     b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
#     return b

# # ==== TRANSLATION ====
# def load_translation_model(model_name, device):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
#     return tokenizer, model

# def translate_sentence(tokenizer, model, sentence, device):
#     prompt = f"Translate the following English text to Vietnamese:\n{sentence}"
#     messages = [{"role": "user", "content": prompt}]
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
#     inputs = tokenizer([text], return_tensors="pt").to(device)
#     with torch.no_grad():
#         output_ids = model.generate(**inputs, max_new_tokens=1024)[0][len(inputs.input_ids[0]):].tolist()
#     try:
#         idx = len(output_ids) - output_ids[::-1].index(151668)  # </think>
#     except ValueError:
#         idx = 0
#     translation = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
#     return translation

# # ==== MAIN ====
# def main():
#     args = get_args()
#     os.makedirs(args.output_dir, exist_ok=True)

#     # Build RelTR model
#     class ArgsRelTR:
#         def __init__(self):
#             self.lr_backbone = 1e-5
#             self.dataset = 'vg'
#             self.backbone = 'resnet50'
#             self.dilation = False
#             self.position_embedding = 'sine'
#             self.enc_layers = 6
#             self.dec_layers = 6
#             self.dim_feedforward = 2048
#             self.hidden_dim = 256
#             self.dropout = 0.1
#             self.nheads = 8
#             self.num_entities = 100
#             self.num_triplets = 200
#             self.pre_norm = False
#             self.aux_loss = True
#             self.device = args.device
#             self.resume = args.checkpoint
#             self.set_cost_class = 1
#             self.set_cost_bbox = 5
#             self.set_cost_giou = 2
#             self.set_iou_threshold = 0.7
#             self.bbox_loss_coef = 5
#             self.giou_loss_coef = 2
#             self.rel_loss_coef = 1
#             self.eos_coef = 0.1
#             self.return_interm_layers = False

#     reltr_args = ArgsRelTR()
#     model_reltr, _, _ = build_model(reltr_args)
#     checkpoint = torch.load(reltr_args.resume, map_location=args.device, weights_only=False)
#     model_reltr.load_state_dict(checkpoint['model'])
#     model_reltr.to(args.device)
#     model_reltr.eval()

#     # Translation model
#     tokenizer_trans, model_trans = load_translation_model(args.translation_model, args.device)

#     # Image transform
#     transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])])

#     results_all = []
#     img_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
#     print(f"Processing {len(img_files)} images...")

#     for img_file in img_files:
#         img_path = os.path.join(args.img_dir, img_file)
#         try:
#             im = Image.open(img_path).convert("RGB")
#         except:
#             print(f"Cannot open {img_file}, skipping.")
#             continue

#         img_tensor = transform(im).unsqueeze(0).to(args.device)
#         with torch.no_grad():
#             outputs = model_reltr(img_tensor)

#         probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
#         probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
#         probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]

#         keep = torch.logical_and(probas.max(-1).values>args.threshold,
#                                  torch.logical_and(probas_sub.max(-1).values>args.threshold,
#                                                    probas_obj.max(-1).values>args.threshold))

#         topk = args.topk
#         keep_queries = torch.nonzero(keep, as_tuple=True)[0]
#         scores = probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0]
#         _, indices = scores.topk(min(topk, len(scores)))
#         keep_queries = keep_queries[indices]

#         seen_rel = set()
#         scene_graph = []

#         for idx in keep_queries:
#             subj = CLASSES[probas_sub[idx].argmax()]
#             pred = REL_CLASSES[probas[idx].argmax()]
#             obj = CLASSES[probas_obj[idx].argmax()]
#             score = float(probas[idx].max().item())

#             key = f"{subj}_{pred}_{obj}"
#             if key in seen_rel:
#                 continue
#             seen_rel.add(key)

#             eng_sentence = f"{subj} {pred} {obj}"
#             try:
#                 translation_vi = translate_sentence(tokenizer_trans, model_trans, eng_sentence, args.device)
#             except:
#                 translation_vi = eng_sentence

#             scene_graph.append({
#                 "subject_en": subj,
#                 "predicate_en": pred,
#                 "object_en": obj,
#                 "translated_vi": translation_vi,
#                 "score": round(score, 4)
#             })

#         results_all.append({"file_name": img_file, "scene_graph": scene_graph})
#         print(f"Processed {img_file}: {len(scene_graph)} relations.")

#     # Save JSON
#     with open(os.path.join(args.output_dir, "scene_graphs_translated.json"), "w", encoding="utf-8") as f:
#         json.dump(results_all, f, ensure_ascii=False, indent=2)

#     print(f"Done. Results saved in {args.output_dir}/scene_graphs_translated.json")

# if __name__ == "__main__":
#     main()


import os
import json
from PIL import Image
import torch
import torchvision.transforms as T
from models import build_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# ==== ARGPARSE ====
def get_args():
    parser = argparse.ArgumentParser(description="RelTR inference + translate scene graph via LLM")
    parser.add_argument('--img_dir', type=str, required=True, help="Folder of test images")
    parser.add_argument('--output_dir', type=str, default="outputs_reltr", help="Folder to save JSON results")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to RelTR checkpoint")
    parser.add_argument('--translation_model', type=str, default="Qwen/Qwen3-8B", help="HuggingFace model for translation")
    parser.add_argument('--threshold', type=float, default=0.25, help="Confidence threshold for keeping relations")
    parser.add_argument('--topk', type=int, default=10, help="Top-K relations per image")
    return parser.parse_args()

# ==== LABELS ====
CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
               'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
               'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
               'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
               'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
               'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

# ==== UTILS ====
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

# ==== TRANSLATION ====
def load_translation_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    return tokenizer, model

def translate_and_split(tokenizer, model, sentence):
    """
    Translate English sentence to Vietnamese and split into subject, predicate, object
    """
    prompt = f"""
    Translate the following English sentence into Vietnamese, and separate it into Subject, Predicate, and Object.
    Format your answer as JSON with keys: "subject_vi", "predicate_vi", "object_vi".

    Sentence: "{sentence}"
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    out_text = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    try:
        parsed = json.loads(out_text)
        return parsed.get("subject_vi",""), parsed.get("predicate_vi",""), parsed.get("object_vi","")
    except json.JSONDecodeError:
        return out_text, "", ""

# ==== MAIN ====
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Build RelTR
    class ArgsRelTR:
        def __init__(self):
            self.lr_backbone = 1e-5
            self.dataset = 'vg'
            self.backbone = 'resnet50'
            self.dilation = False
            self.position_embedding = 'sine'
            self.enc_layers = 6
            self.dec_layers = 6
            self.dim_feedforward = 2048
            self.hidden_dim = 256
            self.dropout = 0.1
            self.nheads = 8
            self.num_entities = 100
            self.num_triplets = 200
            self.pre_norm = False
            self.aux_loss = True
            self.device = args.device
            self.resume = args.checkpoint
            self.set_cost_class = 1
            self.set_cost_bbox = 5
            self.set_cost_giou = 2
            self.set_iou_threshold = 0.7
            self.bbox_loss_coef = 5
            self.giou_loss_coef = 2
            self.rel_loss_coef = 1
            self.eos_coef = 0.1
            self.return_interm_layers = False

    reltr_args = ArgsRelTR()
    model_reltr, _, _ = build_model(reltr_args)
    checkpoint = torch.load(reltr_args.resume, map_location=args.device, weights_only=False)
    model_reltr.load_state_dict(checkpoint['model'])
    model_reltr.to(args.device)
    model_reltr.eval()

    # Load LLM translation model
    tokenizer_trans, model_trans = load_translation_model(args.translation_model, args.device)

    # Image transform
    transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    results_all = []
    img_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"Processing {len(img_files)} images...")

    for img_file in img_files:
        img_path = os.path.join(args.img_dir, img_file)
        try:
            im = Image.open(img_path).convert("RGB")
        except:
            print(f"Cannot open {img_file}, skipping.")
            continue

        img_tensor = transform(im).unsqueeze(0).to(args.device)
        with torch.no_grad():
            outputs = model_reltr(img_tensor)

        probas = outputs['rel_logits'].softmax(-1)[0,:,:-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0,:,:-1]
        probas_obj = outputs['obj_logits'].softmax(-1)[0,:,:-1]

        keep = torch.logical_and(
            probas.max(-1).values>args.threshold,
            torch.logical_and(probas_sub.max(-1).values>args.threshold, probas_obj.max(-1).values>args.threshold)
        )

        keep_queries = torch.nonzero(keep, as_tuple=True)[0]
        scores = probas[keep_queries].max(-1)[0]*probas_sub[keep_queries].max(-1)[0]*probas_obj[keep_queries].max(-1)[0]
        _, indices = scores.topk(min(args.topk, len(scores)))
        keep_queries = keep_queries[indices]

        scene_graph = []
        seen_rel = set()

        for idx in keep_queries:
            subj = CLASSES[probas_sub[idx].argmax()]
            pred = REL_CLASSES[probas[idx].argmax()]
            obj = CLASSES[probas_obj[idx].argmax()]
      
            key = f"{subj}_{pred}_{obj}"
            if key in seen_rel:
                continue
            seen_rel.add(key)

            eng_sentence = f"{subj} {pred} {obj}"
            subj_vi, pred_vi, obj_vi = translate_and_split(tokenizer_trans, model_trans, eng_sentence)

            scene_graph.append({
                "subject_en": subj,
                "predicate_en": pred,
                "object_en": obj,
                "subject_vi": subj_vi,
                "predicate_vi": pred_vi,
                "object_vi": obj_vi,
            })

        results_all.append({"file_name": img_file, "scene_graph": scene_graph})
        print(f"Processed {img_file}: {len(scene_graph)} relations.")

    # Save JSON
    with open(os.path.join(args.output_dir, "scene_graphs_translated_train.json"), "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    print(f"Done. Results saved in {args.output_dir}/scene_graphs_translated_train.json")

if __name__=="__main__":
    main()
