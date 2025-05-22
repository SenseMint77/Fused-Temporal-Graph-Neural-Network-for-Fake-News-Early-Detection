import os
import sys
import time
import random
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import torch.autograd as autogradimport torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import EarlyStopping, \
    compute_classification_metrics
from paths_ko import PathSet, \
    PathSetBERT
from data import split_data_stratified
from config import BaseConfig, DualGraphContextMAGNNConfig, \
    DualGraphContextMAGNNConfigKO
from models import DualGraphRumorModelBase, \
    CompletedGRUWithoutBERTModel
from data_process import GraphDataProcessor
from torch_geometric.loader import \
    DataLoader as PyGDataLoader
import warnings

warnings.filterwarnings('ignore')  # 경고 메시지 무시 (주의해서 사용)


def train_and_evaluate_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        training_config,
        cli_args
):
    """
    Handles the training and validation loop for the model.
    """
    model.to(device)  # Ensure model is on the correct device

    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer setup
    # If BERT is used and not frozen, separate learning rates can be set for BERT and other parts of the model
    if hasattr(model, 'bert_model') and model.bert_model is not None and \
            any(p.requires_grad for p in model.bert_model.parameters()):

        bert_params = list(model.bert_model.parameters())
        bert_param_ids = {id(p) for p in bert_params}
        base_model_params = [p for p in model.parameters() if id(p) not in bert_param_ids and p.requires_grad]

        optimizer = optim.AdamW([
            {'params': model.bert_model.parameters(), 'lr': getattr(cli_args, 'bert_lr', 2e-5)},
            {'params': base_model_params, 'lr': cli_args.lr}
        ], weight_decay=training_config.weight_decay)
        print("Optimizer: Configured with separate learning rates for BERT and base model.")
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cli_args.lr, weight_decay=training_config.weight_decay)
        print("Optimizer: Configured with a single learning rate for all trainable parameters.")

    # Loss function
    if training_config.n_class == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5,
                                                     verbose=True)  # 검증 손실 모니터링 (Monitor validation loss)

    # Early stopping
    early_stopper = EarlyStopping(patience=training_config.patience, verbose=True,
                                  model_save_path=cli_args.model_save_dir,
                                  model_name=f"{cli_args.dataset}_{cli_args.model}_best.pt",
                                  monitor_metric_name="val_accuracy" if cli_args.early_stop_metric == "acc" else "val_loss")

    best_val_metric_value = -np.inf if cli_args.early_stop_metric == "acc" else np.inf
    best_test_results_at_best_val = {}

    start_time_train = time.time()

    for epoch in range(cli_args.epoch):
        print(f"\n--- 에폭 {epoch + 1}/{cli_args.epoch} ---")
        #  Training Phase
        model.train()
        total_train_loss = 0
        all_train_preds, all_train_targets = [], []

        progress_bar_train = tqdm(train_loader, desc=f"에폭 {epoch + 1} 학습 중 (Epoch {epoch + 1} Training)", leave=False)
        for batch_data in progress_bar_train:
            optimizer.zero_grad()

            outputs = model(batch_data)
            target_y = batch_data.y.to(device) if hasattr(batch_data, 'y') else batch_data['post'].y.to(
                device)  # 예시

            if training_config.n_class == 1:
                loss = criterion(outputs.squeeze(), target_y.float())
            else:
                loss = criterion(outputs, target_y.long().squeeze())

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            all_train_preds.append(outputs.detach())
            all_train_targets.append(target_y.detach())
            progress_bar_train.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_preds_all = torch.cat(all_train_preds)
        train_targets_all = torch.cat(all_train_targets)
        train_metrics = compute_classification_metrics(train_preds_all, train_targets_all,
                                                       num_classes=training_config.n_class)
        print(
            f"에폭 {epoch + 1} 학습: 손실={avg_train_loss:.4f}, 정확도={train_metrics['accuracy']:.4f}, F1-매크로={train_metrics.get('f1_macro', 0):.4f}")
           # --- Validation ---
        model.eval()
        total_val_loss = 0
        all_val_preds, all_val_targets = [], []

        progress_bar_val = tqdm(val_loader, desc=f"에폭 {epoch + 1} 검증 중 (Epoch {epoch + 1} Validation)", leave=False)
        with torch.no_grad():
            for batch_data in progress_bar_val:
                # batch_data = batch_data.to(device)
                outputs = model(batch_data)
                target_y = batch_data.y.to(device) if hasattr(batch_data, 'y') else batch_data['post'].y.to(
                    device)  # 예시

                if training_config.n_class == 1:
                    loss = criterion(outputs.squeeze(), target_y.float())
                else:
                    loss = criterion(outputs, target_y.long().squeeze())

                total_val_loss += loss.item()
                all_val_preds.append(outputs.detach())
                all_val_targets.append(target_y.detach())
                progress_bar_val.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_preds_all = torch.cat(all_val_preds)
        val_targets_all = torch.cat(all_val_targets)
        val_metrics = compute_classification_metrics(val_preds_all, val_targets_all,
                                                     num_classes=training_config.n_class)
        print(
            f"(Epoch {epoch+1} Validation: Loss={avg_val_loss:.4f}, Accuracy={val_metrics['accuracy']:.4f}, F1-Macro={val_metrics.get('f1_macro', 0):.4f})")
        scheduler.step(avg_val_loss)

        current_monitored_metric = val_metrics['accuracy'] if cli_args.early_stop_metric == "acc" else avg_val_loss
        higher_is_better_metric = True if cli_args.early_stop_metric == "acc" else False

        if early_stopper(current_monitored_metric, model, higher_is_better=higher_is_better_metric):
             if early_stopper.best_score == (
            current_monitored_metric if higher_is_better_metric else -current_monitored_metric):  # 점수가 실제로 개선된 경우 (If score actually improved)
                print(f"Best validation {cli_args.early_stop_metric}: {early_stopper.best_metric_value:.4f}. Evaluating on test set.")

                best_test_results_at_best_val = test_model(test_loader, model, device, training_config,
                                                           "Validation Checkpoint")

        if early_stopper.early_stop_triggered:
            print("Early stopping triggered.")  # ()
            break

    end_time_train = time.time()
    print(f"\nTraining finished. Total time:{(end_time_train - start_time_train) / 60:.2f} minutes.")

    print("\nLoading best model for final test evaluation...)")
    model = early_stopper.load_best_weights(model)

    if not best_test_results_at_best_val:
        print("마지막 에폭의 모델(또는 개선 없는 경우 초기 모델)로 테스트 세트에서 평가 중.")
        # ()
        best_test_results_at_best_val = test_model(test_loader, model, device, training_config, "최종 모델 (Final Model)")

    print("\n--- 최종 최고 테스트 결과 (최고 검증 에폭 기준) ---")
    # (--- Final Best Test Results (from best validation epoch) ---)
    for metric, value in best_test_results_at_best_val.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return best_test_results_at_best_val, start_time_train, end_time_train


def test_model(test_loader, model, device, testing_config, phase_name="테스트 (Test)"):
    """테스트 세트에서 모델을 평가합니다. (Evaluates the model on the test set.)"""
    model.eval()
    all_test_preds, all_test_targets = [], []

    criterion = nn.BCEWithLogitsLoss() if testing_config.n_class == 1 else nn.CrossEntropyLoss()
    total_test_loss = 0

    progress_bar_test = tqdm(test_loader, desc=f"{phase_name} 평가 중 (Evaluation)", leave=False)
    with torch.no_grad():
        for batch_data in progress_bar_test:
            # batch_data = batch_data.to(device)
            outputs = model(batch_data)
            target_y = batch_data.y.to(device) if hasattr(batch_data, 'y') else batch_data['post'].y.to(
                device)  # 예시 (Example)

            if testing_config.n_class == 1:
                loss = criterion(outputs.squeeze(), target_y.float())
            else:
                loss = criterion(outputs, target_y.long().squeeze())
            total_test_loss += loss.item()

            all_test_preds.append(outputs.detach())
            all_test_targets.append(target_y.detach())

    avg_test_loss = total_test_loss / len(test_loader)
    test_preds_all = torch.cat(all_test_preds)
    test_targets_all = torch.cat(all_test_targets)
    test_metrics = compute_classification_metrics(test_preds_all, test_targets_all, num_classes=testing_config.n_class)

    print(f"\n--- {phase_name} 결과 (Results) ---")
    print(f"손실 (Loss): {avg_test_loss:.4f}")
    for metric, value in test_metrics.items():
        if isinstance(value, float): print(f"{metric}: {value:.4f}")
    return test_metrics


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")
    print(f"사용 장치 (Using device): {device}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 설정 (Configuration) ---
    # dataset 인자에 따라 언어 설정 및 기본 config 선택 (Language setting and default config selection based on dataset argument)
    language = 'korean' if 'ko' in args.dataset.lower() or 'weibo' in args.dataset.lower() else 'english'  # 데이터셋 이름 기반 언어 추론 (Infer language based on dataset name)
    if args.dataset == 'pheme_ko':  # 한국어 버전 Pheme 데이터셋 예시 (Korean version Pheme dataset example)
        app_config = DualGraphContextMAGNNConfigKO()  # 한국어용 설정 (Config for Korean)
    elif args.dataset == 'weibo_ko':  # 한국어 버전 Weibo 데이터셋 예시 (Korean version Weibo dataset example)
        app_config = DualGraphContextMAGNNConfigKO()
    elif args.dataset == 'pheme':  # 영어 Pheme (English Pheme)
        app_config = DualGraphContextMAGNNConfig()
        language = 'english'
    else:  # 기타 또는 기본 (Other or default)
        print(f"경고: 데이터셋 '{args.dataset}'에 대한 특정 설정이 없습니다. 기본 설정을 사용합니다.")
        # (Warning: No specific configuration for dataset '{args.dataset}'. Using default settings.)
        app_config = DualGraphContextMAGNNConfig()  # 기본 영어 설정 또는 일반 설정 (Default English or general settings)
        language = 'english'  # 기본값 (Default value)

    app_config.text_max_length = args.text_max_len if args.text_max_len else app_config.text_max_length
    # MAGNN 관련 설정값도 args에서 받아와 app_config에 설정 가능 (MAGNN related settings can also be taken from args and set in app_config)
    app_config.magnn_num_metapaths = args.magnn_metapaths if hasattr(args,
                                                                     'magnn_metapaths') else app_config.magnn_num_metapaths

    # --- 경로 설정 (Path Configuration) ---
    path_manager = PathSetBERT(args.dataset,
                               bert_base_dir=args.bert_dir)  # BERT 사용 여부와 관계없이 BERT 경로 설정 포함 가능 (Can include BERT path settings regardless of whether BERT is used)
    if hasattr(app_config, 'use_bert_for_root') and app_config.use_bert_for_root:
        app_config.bert_model_name_or_path = path_manager.bert_model_name_or_path

    # --- 데이터 처리 (Data Processing) ---
    # GraphDataProcessor는 HIN/MAGNN을 위해 HeteroData 객체를 생성해야 함 (GraphDataProcessor needs to create HeteroData objects for HIN/MAGNN)
    use_bert_for_node_text = True if "bert" in args.model.lower() and "wobert" not in args.model.lower() else False

    data_processor = GraphDataProcessor(
        path_config=path_manager,
        app_config=app_config,  # 전체 app_config 전달 (Pass entire app_config)
        language=language
    )

    # HeteroData 객체 리스트 로드 (load_and_prepare_data_for_magnn는 상세 구현 필요)
    # (Load list of HeteroData objects (load_and_prepare_data_for_magnn needs detailed implementation))
    train_pyg_data, val_pyg_data, test_pyg_data = data_processor.load_and_prepare_data(
        app_config.train_ratio, app_config.val_ratio, app_config.test_ratio
    )
    if not train_pyg_data:  # 데이터 로딩 실패 시 (If data loading fails)
        print("데이터 로드 또는 처리에 실패했습니다. 종료합니다.")  # (Failed to load or process data. Exiting.)
        return

    train_loader = PyGDataLoader(train_pyg_data, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)
    val_loader = PyGDataLoader(val_pyg_data, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    test_loader = PyGDataLoader(test_pyg_data, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    print(f"데이터 로드 완료: 학습 {len(train_pyg_data)} 샘플, 검증 {len(val_pyg_data)}, 테스트 {len(test_pyg_data)}")
    # (Data loaded: Train {len(train_pyg_data)} samples, Validation {len(val_pyg_data)}, Test {len(test_pyg_data)})

    # --- 모델 선택 및 초기화 (Model Selection & Initialization) ---
    text_vocab_size = len(data_processor.token_to_id_map) if not use_bert_for_node_text else 0
    # HIN의 경우, graph_node_vocab_size는 타입별 어휘 크기 또는 전역 ID 공간 크기를 의미할 수 있음
    # (For HIN, graph_node_vocab_size can mean vocabulary size by type or global ID space size)
    # data_processor.node_to_global_idx 사용 (Use data_processor.node_to_global_idx)
    graph_node_vocab_size = data_processor.next_global_idx

    # args.model 값에 따라 적절한 모델 클래스 선택 및 설정 (Select and set appropriate model class based on args.model value)
    if args.model == 'completed_gru_wobert_magnn':  # MAGNN 사용하는 GRU 기반 모델 예시 (Example of GRU-based model using MAGNN)
        app_config.use_bert_for_root = False
        app_config.concat_root_bert_to_pooled = False
        # CompletedGRUWithoutBERTModel을 MAGNN에 맞게 수정하거나, 새로운 MAGNN용 모델 클래스 필요
        # (CompletedGRUWithoutBERTModel needs to be modified for MAGNN, or a new model class for MAGNN is needed)
        # 여기서는 CompletedGRUWithoutBERTModel이 MAGNN 입력을 처리할 수 있도록 _prepare_initial_node_features가 잘 구현되었다고 가정
        # (Here, assuming _prepare_initial_node_features is well implemented so that CompletedGRUWithoutBERTModel can handle MAGNN input)
        model = CompletedGRUWithoutBERTModel(
            config=app_config,
            text_vocab_size=text_vocab_size,
            graph_node_vocab_size=graph_node_vocab_size,
            device=device,
            mid_to_token_ids_map=data_processor.nodeidx_to_token_ids if not use_bert_for_node_text else {}
            # 또는 BERT 입력 맵 (Or BERT input map)
        )
    # elif args.model == '다른_MAGNN_모델': (elif args.model == 'other_MAGNN_model':)
    #    model = 다른MAGNN모델클래스(app_config, ..., device) (model = OtherMAGNNModelClass(app_config, ..., device))
    else:
        print(f"경고: 모델 타입 '{args.model}'이 명시적으로 구성되지 않았습니다. 플레이스홀더 모델을 사용합니다.")
        # (Warning: Model type '{args.model}' not explicitly configured. Using placeholder model.)
        model = CompletedGRUWithoutBERTModel(app_config, text_vocab_size, graph_node_vocab_size, device,
                                             data_processor.nodeidx_to_token_ids if not use_bert_for_node_text else {})

    # --- 학습 및 평가 (Training & Evaluation) ---
    best_test_results, total_start_time, total_end_time = train_and_evaluate_model(
        model, train_loader, val_loader, test_loader, device, app_config, args
    )

    print(f"\n--- 전체 최고 테스트 성능 ({args.dataset} 데이터셋, {args.model} 모델, 최고 검증 에폭 기준) ---")
    # (--- Overall Best Test Performance ({args.dataset} dataset, {args.model} model, based on best validation epoch) ---)
    print(f"파라미터: 배치 크기={args.batch}, 에폭 수={args.epoch}, 학습률={args.lr}, BERT 학습률={getattr(args, 'bert_lr', '해당 없음')}")
    # (Parameters: Batch Size={args.batch}, Epochs={args.epoch}, LR={args.lr}, BERT LR={getattr(args, 'bert_lr', 'N/A')})
    for metric, value in best_test_results.items():
        if isinstance(value, float): print(f"{metric}: {value:.4f}")
    print(f"총 스크립트 실행 시간: {(total_end_time - total_start_time) / 60:.2f} 분.")
    # (Total script execution time: {(total_end_time - total_start_time) / 60:.2f} minutes.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='듀얼 동적 그래프 컨볼루션 네트워크 (문맥 그래프 MAGNN 적용) 루머 탐지 (Dual Dynamic Graph Convolutional Networks (Context Graph MAGNN Applied) for Rumor Detection)')
    parser.add_argument('--dataset', type=str, default='pheme_ko', choices=['pheme_ko', 'weibo_ko', 'pheme', 'weibo'],
                        help='사용할 데이터셋. (Dataset to use.)')
    parser.add_argument('--model', type=str, default='completed_gru_wobert_magnn',
                        help='실행할 모델 변형 (예: completed_gru_wobert_magnn) (Model variant to run (e.g., completed_gru_wobert_magnn))')
    parser.add_argument('--cuda', type=int, default=0, help='사용할 GPU ID. CPU의 경우 -1. (GPU ID to use. -1 for CPU.)')
    parser.add_argument('--batch', type=int, default=16,
                        help='학습 및 평가 배치 크기. (Batch size for training and evaluation.)')
    parser.add_argument('--epoch', type=int, default=50, help='학습 에폭 수. (Number of training epochs.)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='모델의 비-BERT 부분 학습률. (Learning rate for non-BERT parts of the model.)')
    parser.add_argument('--bert_lr', type=float, default=2e-5,
                        help='모델의 BERT 부분 학습률 (사용 및 동결 해제 시). (Learning rate for BERT parts of the model (if used and unfrozen).)')
    parser.add_argument('--text_max_len', type=int, default=None,
                        help='텍스트 시퀀스 최대 길이 (config 재정의). (Maximum length for text sequences (overrides config).)')
    parser.add_argument('--seed', type=int, default=42, help='재현성을 위한 랜덤 시드. (Random seed for reproducibility.)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader 워커 수. 0은 메인 프로세스 사용. (Number of workers for DataLoader. 0 for main process.)')
    parser.add_argument('--model_save_dir', type=str, default="trained_models_output_magnn",
                        help='최상 모델 저장 디렉토리. (Directory to save best models.)')
    parser.add_argument('--bert_dir', type=str, default="bert_models/",
                        help='사전 학습된 BERT 모델 기본 디렉토리. (Base directory for pre-trained BERT models.)')
    parser.add_argument('--early_stop_metric', type=str, default="acc", choices=['acc', 'loss'],
                        help='조기 종료용 지표 (val_accuracy 또는 val_loss). (Metric for early stopping (val_accuracy or val_loss).)')
    parser.add_argument('--magnn_metapaths', type=int, default=3,
                        help='MAGNN에서 사용할 메타경로 수. (Number of metapaths to use in MAGNN.)')

    args_cli = parser.parse_args()

    if not os.path.exists(args_cli.model_save_dir):
        os.makedirs(args_cli.model_save_dir, exist_ok=True)

    main(args_cli)
