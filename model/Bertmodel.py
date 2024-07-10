import os
import math
import numpy as np
import pandas as pd
import torch
import logging
import random
import warnings
import pkg_resources
import sklearn
import nni
from torch.nn import MSELoss
from tqdm.auto import tqdm, trange
from itertools import zip_longest
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from rdkit import Chem
from simpletransformers.losses.loss_utils import init_loss

from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from simpletransformers.classification.classification_utils import (
    InputExample,
    LazyClassificationDataset,
    ClassificationDataset,
    convert_examples_to_features,
    load_hf_dataset,
    flatten_results,
)
from transformers import (
    BertConfig, BertForMaskedLM, AlbertConfig, AlbertForMaskedLM
)

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

from .tokenizer import SmilesTokenizer
logger = logging.getLogger(__name__)

try:
    import simpletransformers
    # The original results were obtained with simpletransformers==0.34.4 and transformers==2.11.0")
except ImportError:
    raise ImportError('To use this extension, please install simpletransformers ( "pip install simpletransformers==0.61.13"')


# Cell
# optional
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import ClassificationArgs
from simpletransformers.classification import ClassificationModel

from simpletransformers.classification.classification_model import (MODELS_WITHOUT_SLIDING_WINDOW_SUPPORT,
                                               MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT,
                                               MODELS_WITH_EXTRA_SEP_TOKEN,
                                               MODELS_WITH_ADD_PREFIX_SPACE)
from transformers import BertForSequenceClassification

class SmilesClassificationModel(ClassificationModel):
    def __init__(
        self,
        model_type,
        model_name,
        tokenizer_type=None,
        tokenizer_name=None,
        num_labels=None,
        weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        onnx_execution_provider=None,
        freeze_encoder=False,
        freeze_all_but_one=False,
        hte_w=0.3,
        lit_w=0.7,
        **kwargs,
    ):
        self.train_dataloader_hte=None,
        self.train_dataloader_lit=None,
        self.hte_w = hte_w
        self.lit_w = lit_w
        MODEL_CLASSES = {
            "bert": (BertConfig, BertForSequenceClassification, SmilesTokenizer),
        }

        if model_type not in MODEL_CLASSES.keys():
            raise NotImplementedException(f"Currently the following model types are implemented: {MODEL_CLASSES.keys()}")

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, ClassificationArgs):
            self.args = args

        if (
            model_type in MODELS_WITHOUT_SLIDING_WINDOW_SUPPORT
            and self.args.sliding_window
        ):
            raise ValueError(
                "{} does not currently support sliding window".format(model_type)
            )

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)
        self.is_sweeping = False
        torch.backends.cudnn.benchmark = True
        
        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.backends.cudnn.benchmark = False
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if self.args.labels_list:
            if num_labels:
                assert num_labels == len(self.args.labels_list)
            if self.args.labels_map:
                try:
                    assert list(self.args.labels_map.keys()) == self.args.labels_list
                except AssertionError:
                    assert [
                        int(key) for key in list(self.args.labels_map.keys())
                    ] == self.args.labels_list
                    self.args.labels_map = {
                        int(key): value for key, value in self.args.labels_map.items()
                    }
            else:
                self.args.labels_map = {
                    label: i for i, label in enumerate(self.args.labels_list)
                }
        else:
            len_labels_list = 2 if not num_labels else num_labels
            self.args.labels_list = [i for i in range(len_labels_list)]

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if tokenizer_type is not None:
            if isinstance(tokenizer_type, str):
                _, _, tokenizer_class = MODEL_CLASSES[tokenizer_type]
            else:
                tokenizer_class = tokenizer_type

        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **self.args.config
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels

        if model_type in MODELS_WITHOUT_CLASS_WEIGHTS_SUPPORT and weight is not None:
            raise ValueError(
                "{} does not currently support class weights".format(model_type)
            )
        else:
            self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"

        self.loss_fct = init_loss(
            weight=self.weight, device=self.device, args=self.args
        )

        if self.args.onnx:
            from onnxruntime import InferenceSession, SessionOptions

            if not onnx_execution_provider:
                onnx_execution_provider = (
                    "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
                )

            options = SessionOptions()

            if self.args.dynamic_quantize:
                model_path = quantize(Path(os.path.join(model_name, "onnx_model.onnx")))
                self.model = InferenceSession(
                    model_path.as_posix(), options, providers=[onnx_execution_provider]
                )
            else:
                model_path = os.path.join(model_name, "onnx_model.onnx")
                self.model = InferenceSession(
                    model_path, options, providers=[onnx_execution_provider]
                )
        else:
            if not self.args.quantized_model:
                if self.weight:
                    self.model = model_class.from_pretrained(
                        model_name,
                        config=self.config,
                        weight=torch.Tensor(self.weight).to(self.device, non_blocking=True),
                        **kwargs,
                    )
                else:
                    self.model = model_class.from_pretrained(
                        model_name, config=self.config, **kwargs
                    )
            else:
                quantized_weights = torch.load(
                    os.path.join(model_name, "pytorch_model.bin")
                )
                if self.weight:
                    self.model = model_class.from_pretrained(
                        None,
                        config=self.config,
                        state_dict=quantized_weights,
                        weight=torch.Tensor(self.weight).to(self.device, non_blocking=True),
                    )
                else:
                    self.model = model_class.from_pretrained(
                        None, config=self.config, state_dict=quantized_weights
                    )

            if self.args.dynamic_quantize:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            if self.args.quantized_model:
                self.model.load_state_dict(quantized_weights)
            if self.args.dynamic_quantize:
                self.args.quantized_model = True

        self.results = {}

        self.tokenizer = SmilesTokenizer("data/vocab.txt", do_lower_case=False)

        self.args.model_name = model_name
        self.args.model_type = model_type
        self.args.tokenizer_name = tokenizer_name
        self.args.tokenizer_type = tokenizer_type

        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    continue
                param.requires_grad = False
        elif freeze_all_but_one:
            n_layers = self.model.config.num_hidden_layers
            for name, param in self.model.named_parameters():
                if str(n_layers-1) in name:
                    continue
                elif 'classifier' in name:
                    continue
                elif 'pooler' in name:
                    continue
                param.requires_grad = False

    def train_model(
        self,
        train_df,
        train_df_hte=None,
        train_df_lit=None,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        test_df=None,
        verbose=True,
        default_metric="",
        std:float=None,
        mean:float=None,
        **kwargs,
    ):

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if std is not None:
            self.args.std = std

        if mean is not None:
            self.args.mean = mean

        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not self.args.overwrite_output_dir
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(
                    output_dir
                )
            )
        self._move_model_to_device()

        if self.args.use_hf_datasets:
            if self.args.sliding_window:
                raise ValueError(
                    "HuggingFace Datasets cannot be used with sliding window."
                )
            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                raise NotImplementedError(
                    "HuggingFace Datasets support is not implemented for LayoutLM models"
                )
            train_dataset = load_hf_dataset(
                train_df, self.tokenizer, self.args, multi_label=multi_label
            )
        elif isinstance(train_df, str) and self.args.lazy_loading:
            if self.args.sliding_window:
                raise ValueError("Lazy loading cannot be used with sliding window.")
            if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                raise NotImplementedError(
                    "Lazy loading is not implemented for LayoutLM models"
                )
            train_dataset = LazyClassificationDataset(
                train_df, self.tokenizer, self.args
            )
        else:
            if self.args.lazy_loading:
                raise ValueError(
                    "Input must be given as a path to a file when using lazy loading"
                )
            if "text" in train_df.columns and "labels" in train_df.columns:
                if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                    train_examples = [
                        InputExample(i, text, None, label, x0, y0, x1, y1)
                        for i, (text, label, x0, y0, x1, y1) in enumerate(
                            zip(
                                train_df["text"].astype(str),
                                train_df["labels"],
                                train_df["x0"],
                                train_df["y0"],
                                train_df["x1"],
                                train_df["y1"],
                            )
                        )
                    ]
                else:
                    #このブランチが使われる
                    train_examples = (
                        train_df["text"].astype(str).tolist(),
                        train_df["labels"].tolist(),
                    )
                    
            elif "text_a" in train_df.columns and "text_b" in train_df.columns:
                if self.args.model_type in ["layoutlm", "layoutlmv2"]:
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    train_examples = (
                        train_df["text_a"].astype(str).tolist(),
                        train_df["text_b"].astype(str).tolist(),
                        train_df["labels"].tolist(),
                    )
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                train_examples = (
                    train_df.iloc[:, 0].astype(str).tolist(),
                    train_df.iloc[:, 1].tolist(),
                )
            train_dataset = self.load_and_cache_examples(
                train_examples, verbose=verbose
            )
            
        train_sampler = RandomSampler(train_dataset)

        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            # num_workers=4,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )
        os.makedirs(output_dir, exist_ok=True)
        if train_df_hte is not None and train_df_lit is not None:
            train_examples_hte = (
                            train_df_hte["text"].astype(str).tolist(),
                            train_df_hte["labels"].tolist(),
            )
            train_examples_lit = (
                train_df_lit["text"].astype(str).tolist(),
                train_df_lit["labels"].tolist(),
            )
            train_dataset_hte = self.load_and_cache_examples(
                    train_examples_hte, verbose=verbose
                )
            train_dataset_lit = self.load_and_cache_examples(
                    train_examples_lit, verbose=verbose
                )
            train_sampler_hte = RandomSampler(train_dataset_hte)
            train_sampler_lit = RandomSampler(train_dataset_lit)
            self.train_dataloader_hte = DataLoader(
            train_dataset_hte,
            sampler=train_sampler_hte,
            batch_size=self.args.train_batch_size,
            # num_workers=4,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
            )
            self.train_dataloader_lit = DataLoader(
            train_dataset_lit,
            sampler=train_sampler_lit,
            batch_size=self.args.train_batch_size,
            # num_workers=4,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
            )
            global_step, training_details = self.train(
                train_dataloader,
                True,
                output_dir,
                multi_label=multi_label,
                show_running_loss=show_running_loss,
                eval_df=eval_df,
                test_df=test_df,
                verbose=verbose,
                default_metric=default_metric,
                **kwargs,
            )
        else:
            global_step, training_details = self.train(
                train_dataloader,
                False,
                output_dir,
                multi_label=multi_label,
                show_running_loss=show_running_loss,
                eval_df=eval_df,
                test_df=test_df,
                verbose=verbose,
                default_metric=default_metric,
                **kwargs,
            )
        self.save_model(model=self.model)

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_type, output_dir
                )
            )

        return global_step, training_details
                
    def train(
        self,
        train_dataloader,
        is_weighted,
        output_dir,
        multi_label=False,
        show_running_loss=True,
        eval_df=None,
        test_df=None,
        verbose=True,
        default_metric="",
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args


        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
                betas=args.adam_betas,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )

        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )
        
        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )
        
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad(set_to_none=True)
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        current_loss = "Initializing"


        
        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                multi_label, **kwargs
            )
        

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            if is_weighted:
                zipped_dataloaders_longest = zip_longest(self.train_dataloader_hte, self.train_dataloader_lit, fillvalue=None)
                batch_iterator = tqdm(
                    zipped_dataloaders_longest,
                    desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                    disable=args.silent,
                    mininterval=0,
                )
                for step, (batch1, batch2) in enumerate(batch_iterator):
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue
                    if batch2 is None:
                        inputs1 = self._get_inputs_dict(batch1)
                        inputs2 = None
                        inputs = inputs1.copy()
                    else:
                        inputs1 = self._get_inputs_dict(batch1)
                        inputs2 = self._get_inputs_dict(batch2)
                        inputs = inputs1.copy()
                        inputs.update(inputs2)
                    if self.args.fp16:
                        with amp.autocast():
                            loss, *_ = self._calculate_loss_wieghed(
                                model,
                                inputs1,
                                inputs2,
                                loss_fct=self.loss_fct,
                                num_labels=self.num_labels,
                                args=self.args,
                            )
                    else:
                        loss, *_ = self._calculate_loss_wieghed(
                            model,
                            inputs1,
                            inputs2,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )

                    if args.n_gpu > 1:
                        loss = (
                            loss.mean()
                        )  # mean() to average on multi-gpu parallel training

                    current_loss = loss.item()

                    if show_running_loss:
                        batch_iterator.set_description(
                            f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                        )

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if self.args.fp16:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if self.args.fp16:
                            scaler.unscale_(optimizer)
                        if args.optimizer == "AdamW":
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.max_grad_norm
                            )

                        if self.args.fp16:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad(set_to_none=True)
                        global_step += 1

                        if args.save_steps > 0 and global_step % args.save_steps == 0:
                            # Save model checkpoint
                            output_dir_current = os.path.join(
                                output_dir, "checkpoint-{}".format(global_step)
                            )

                            self.save_model(
                                output_dir_current, optimizer, scheduler, model=model
                            )

                        if args.evaluate_during_training and (
                            args.evaluate_during_training_steps > 0
                            and global_step % args.evaluate_during_training_steps == 0
                        ):
                            # Only evaluate when single GPU otherwise metrics may not average well
                            results, _, _ = self.eval_model(
                                eval_df,
                                verbose=verbose and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                wandb_log=False,
                                **kwargs,
                            )

                            output_dir_current = os.path.join(
                                output_dir, "checkpoint-{}".format(global_step)
                            )

                            if args.save_eval_checkpoints:
                                self.save_model(
                                    output_dir_current,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )

                            training_progress_scores["global_step"].append(global_step)
                            training_progress_scores["train_loss"].append(current_loss)
                            for key in results:
                                training_progress_scores[key].append(results[key])
                            
                            if test_df is not None:
                                test_results, _, _ = self.eval_model(
                                    test_df,
                                    verbose=verbose
                                    and args.evaluate_during_training_verbose,
                                    silent=args.evaluate_during_training_silent,
                                    wandb_log=False,
                                    **kwargs,
                                )
                                for key in test_results:
                                    training_progress_scores["test_" + key].append(
                                        test_results[key]
                                    )

                            evals_val = make_metric_report(results, default_metric)
                            nni.report_intermediate_result(evals_val)

                            report = pd.DataFrame(training_progress_scores)
                            report.to_csv(
                                os.path.join(
                                    args.output_dir, "training_progress_scores.csv"
                                ),
                                index=False,
                            )

                            if not best_eval_metric:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                            if best_eval_metric and args.early_stopping_metric_minimize:
                                if (
                                    best_eval_metric - results[args.early_stopping_metric]
                                    > args.early_stopping_delta
                                ):
                                    best_eval_metric = results[args.early_stopping_metric]
                                    self.save_model(
                                        args.best_model_dir,
                                        optimizer,
                                        scheduler,
                                        model=model,
                                        results=results,
                                    )
                                    early_stopping_counter = 0
                                else:
                                    if args.use_early_stopping:
                                        if (
                                            early_stopping_counter
                                            < args.early_stopping_patience
                                        ):
                                            early_stopping_counter += 1
                                            if verbose:
                                                logger.info(
                                                    f" No improvement in {args.early_stopping_metric}"
                                                )
                                                logger.info(
                                                    f" Current step: {early_stopping_counter}"
                                                )
                                                logger.info(
                                                    f" Early stopping patience: {args.early_stopping_patience}"
                                                )
                                        else:
                                            if verbose:
                                                logger.info(
                                                    f" Patience of {args.early_stopping_patience} steps reached"
                                                )
                                                logger.info(" Training terminated.")
                                                train_iterator.close()
                                            return (
                                                global_step,
                                                tr_loss / global_step
                                                if not self.args.evaluate_during_training
                                                else training_progress_scores,
                                            )
                            else:
                                if (
                                    results[args.early_stopping_metric] - best_eval_metric
                                    > args.early_stopping_delta
                                ):
                                    best_eval_metric = results[args.early_stopping_metric]
                                    self.save_model(
                                        args.best_model_dir,
                                        optimizer,
                                        scheduler,
                                        model=model,
                                        results=results,
                                    )
                                    early_stopping_counter = 0
                                else:
                                    if args.use_early_stopping:
                                        if (
                                            early_stopping_counter
                                            < args.early_stopping_patience
                                        ):
                                            early_stopping_counter += 1
                                            if verbose:
                                                logger.info(
                                                    f" No improvement in {args.early_stopping_metric}"
                                                )
                                                logger.info(
                                                    f" Current step: {early_stopping_counter}"
                                                )
                                                logger.info(
                                                    f" Early stopping patience: {args.early_stopping_patience}"
                                                )
                                        else:
                                            if verbose:
                                                logger.info(
                                                    f" Patience of {args.early_stopping_patience} steps reached"
                                                )
                                                logger.info(" Training terminated.")
                                                train_iterator.close()
                                            return (
                                                global_step,
                                                tr_loss / global_step
                                                if not self.args.evaluate_during_training
                                                else training_progress_scores,
                                            )
                            model.train()
            else:
                batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )

            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast():
                        loss, *_ = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                else:
                    loss, *_ = self._calculate_loss(
                        model,
                        inputs,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad(set_to_none=True)
                    global_step += 1

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        self.save_model(
                            output_dir_current, optimizer, scheduler, model=model
                        )

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_df,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            wandb_log=False,
                            **kwargs,
                        )

                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        if args.save_eval_checkpoints:
                            self.save_model(
                                output_dir_current,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        
                        if test_df is not None:
                            test_results, _, _ = self.eval_model(
                                test_df,
                                verbose=verbose
                                and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                wandb_log=False,
                                **kwargs,
                            )
                            for key in test_results:
                                training_progress_scores["test_" + key].append(
                                    test_results[key]
                                )

                        evals_val = make_metric_report(results, default_metric)
                        nni.report_intermediate_result(evals_val)

                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(
                                args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (
                                best_eval_metric - results[args.early_stopping_metric]
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()
            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results, _, _ = self.eval_model(
                    eval_df,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    wandb_log=False,
                    **kwargs,
                )

                self.save_model(
                    output_dir_current, optimizer, scheduler, results=results
                )

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)


                for key in results:
                    training_progress_scores[key].append(results[key])
                
                if test_df is not None:
                    test_results, _, _ = self.eval_model(
                        test_df,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        wandb_log=False,
                        **kwargs,
                    )
                    for key in test_results:
                        training_progress_scores["test_" + key].append(
                            test_results[key]
                        )

                evals_val = make_metric_report(results, default_metric)
                nni.report_intermediate_result(evals_val)

                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def _calculate_loss_wieghed(self, model, inputs1, inputs2, loss_fct, num_labels, args):
        outputs_hte = model(**inputs1)
        loss_hte = outputs_hte[0]
        if inputs2 is not None:
            outputs_lit = model(**inputs2)
            loss_lit = outputs_lit[0]
        else:
            loss_lit = 0
        # model outputs are always tuple in pytorch-transformers (see doc)
        
        
        loss = loss_hte*self.hte_w + loss_lit*self.lit_w
        #loss_fct = MSELoss()
        if loss_fct:
            logits_hte = outputs_hte[1]
            labels_hte = inputs1["labels"]
            loss_hte = self.hte_w*loss_fct(logits_hte.view(-1, num_labels), labels_hte.view(-1))
            if inputs2 is not None:
                logits_lit = outputs_lit[1]
                labels_lit = inputs2["labels"]
                loss_lit = self.lit_w*loss_fct(logits_lit.view(-1, num_labels), labels_lit.view(-1))
            else:
                loss_lit = 0

            loss = loss_hte + loss_lit
        return (loss, *outputs_hte[1:])
def make_metric_report(eval_dict, default_metric):
    eval_dict["default"] = eval_dict[default_metric]
    return eval_dict

def can(x):
    mol = Chem.MolFromSmiles(x)
    return Chem.MolToSmiles(mol)
def return_reaction(row, sub1_column, sub2_column, product_column):
    return f"{can(row[sub1_column])}.{can(row[sub2_column])}>>{can(row[product_column])}"

def BERT_train(df, sub1_column, sub2_column, product_column):
    df['text'] = df.apply(lambda row: return_reaction(row, sub1_column, sub2_column, product_column), axis=1)
    df.fillna(0, inplace=True)
    df['labels']= df['yield']
    train_df = df[df['train_or_test']=='train']
    val_df = df[df['train_or_test']!='train']
    test_df = df[df['train_or_test']=='test']
    mean_hte = train_df.labels.mean()
    std_hte = train_df.labels.std()

    train_df['labels'] = (train_df['labels'] - mean_hte) / std_hte
    val_df['labels'] = (val_df['labels'] - mean_hte) / std_hte
    test_df['labels'] = (test_df['labels'] - mean_hte) / std_hte
    model_args = {
        'num_train_epochs': 15, 'overwrite_output_dir': True,
        'learning_rate': 0.00009659, 'gradient_accumulation_steps': 1,
        'regression': True, "num_labels":1, "fp16": False,
        "evaluate_during_training": False, 'manual_seed': 42,
        "max_seq_length": 300, "train_batch_size": 16,"warmup_ratio": 0.00,
        "config" : { 'hidden_dropout_prob': 0.7987 } 
    }

    model_path =  pkg_resources.resource_filename(
                    "rxnfp",
                    f"models/transformers/bert_pretrained" # change pretrained to ft to start from the other base model
    )

    out = "outputs_dir"
    yield_bert.train_model(train_df, output_dir=out, eval_df=val_df, hte_w=0.2,lit_w=0.8)
    tra = list(train_df.text)
    val = list(val_df.text)
    test = list(test_df.text)
    yield_predicted_test = yield_bert.predict(test)[0]
    yield_predicted_test = yield_predicted_test * std_hte + mean_hte
    yield_true_test = test_df.labels.values
    yield_true_test = yield_true_test * std_hte + mean_hte


    MSE = mean_squared_error(yield_true_test,yield_predicted_test)
    MAE = mean_absolute_error(yield_true_test,yield_predicted_test)
    RMSE = np.sqrt(mean_squared_error(yield_true_test,yield_predicted_test))
    R2 = r2_score(yield_true_test,yield_predicted_test)
    Pearson = pearsonr(yield_true_test,yield_predicted_test)
    Spearman = spearmanr(yield_true_test,yield_predicted_test)

    print("MSE:", MSE, '\n', "MAE:", MAE, '\n',"RMSE:", RMSE, '\n',"R2_score:", R2, '\n', "Pearson:", Pearson, '\n', "Spearman", Spearman)