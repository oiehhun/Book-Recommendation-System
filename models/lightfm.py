import numpy as np
import scipy.sparse as sp
# from ._lightfm_fast import (
#     fit_bpr,
#     fit_warp
# )
from models.csr_matrix import CSRMatrix

#__all__ = ["LightFM"]

CYTHON_DTYPE = np.float32

class LightFM(object):
#LightFM 모델 초기화
    def __init__(self, no_components, learning_rate=0.05, item_alpha=0.0, user_alpha=0.0, learning_schedule="adagrad", random_state=None, epsilon=1e-06, n = 10):
        self.learning_schedule = learning_schedule
        #잠재 특징 벡터 크기 
        self.no_components = no_components
        self.learning_rate = learning_rate
        #유저 및 아이템의 총 수 
        self.epsilon = epsilon
        self.item_alpha = item_alpha
        self.user_alpha = user_alpha
        self.n = n
        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)

        self._reset_state()

    #모델 상태 초기화 
    def _reset_state(self):

        self.item_embeddings = None
        self.item_embedding_gradients = None
        self.item_embedding_momentum = None
        self.item_biases = None
        self.item_bias_gradients = None
        self.item_bias_momentum = None

        self.user_embeddings = None
        self.user_embedding_gradients = None
        self.user_embedding_momentum = None
        self.user_biases = None
        self.user_bias_gradients = None
        self.user_bias_momentum = None
    #모델 초기화 여부 확인 
    def _check_initialized(self):

        for var in (
            self.item_embeddings,
            self.item_embedding_gradients,
            self.item_embedding_momentum,
            self.item_biases,
            self.item_bias_gradients,
            self.item_bias_momentum,
            self.user_embeddings,
            self.user_embedding_gradients,
            self.user_embedding_momentum,
            self.user_biases,
            self.user_bias_gradients,
            self.user_bias_momentum,
        ):

            if var is None:
                raise ValueError(
                    "You must fit the model before " "trying to obtain predictions."
                )
   #모델 파라미터 초기화 
    def _initialize(self, no_components, no_item_features, no_user_features):
        # 아이템 피쳐 초기화
        self.item_embeddings = (
            (self.random_state.rand(no_item_features, no_components) - 0.5)
            / no_components
        ).astype(np.float32)
        self.item_embedding_gradients = np.zeros_like(self.item_embeddings)
        self.item_embedding_momentum = np.zeros_like(self.item_embeddings)
        self.item_biases = np.zeros(no_item_features, dtype=np.float32)
        self.item_bias_gradients = np.zeros_like(self.item_biases)
        self.item_bias_momentum = np.zeros_like(self.item_biases)

        # 유저 피쳐 초기화
        self.user_embeddings = (
            (self.random_state.rand(no_user_features, no_components) - 0.5)
            / no_components
        ).astype(np.float32)
        self.user_embedding_gradients = np.zeros_like(self.user_embeddings)
        self.user_embedding_momentum = np.zeros_like(self.user_embeddings)
        self.user_biases = np.zeros(no_user_features, dtype=np.float32)
        self.user_bias_gradients = np.zeros_like(self.user_biases)
        self.user_bias_momentum = np.zeros_like(self.user_biases)

        if self.learning_schedule == "adagrad":
            self.item_embedding_gradients += 1
            self.item_bias_gradients += 1
            self.user_embedding_gradients += 1
            self.user_bias_gradients += 1

    # 사용자와 아이템에 대한 피처 행렬을 구성하는 함수를 정의
    def _construct_feature_matrics(self, n_users, n_items, user_features, item_features):

        # 사용자 피처가 None인 경우, 사용자 수만큼의 단위 행렬을 생성
        if user_features is None:
            user_features = sp.identity(n_users, dtype=CYTHON_DTYPE, format="csr")
        else:
            # 사용자 피처가 None이 아닌 경우, 해당 피처를 CSR 행렬로 변환
            user_features = sp.csr_matrix(user_features)

        # 아이템 피처가 None인 경우, 아이템 수만큼의 단위 행렬을 생성
        if item_features is None:
            item_features = sp.identity(n_items, dtype=CYTHON_DTYPE, format="csr")
        else:
            # 아이템 피처가 None이 아닌 경우, 해당 피처를 CSR 행렬로 변환
            item_features = sp.csr_matrix(item_features)

        # 사용자 수가 사용자 피처 행의 수보다 많은 경우, 오류를 발생
        if n_users > user_features.shape[0]:
            raise Exception(
                "사용자 피처 행의 수가 사용자 수와 일치하지 않습니다."
            )

        # 아이템 수가 아이템 피처 행의 수보다 많은 경우, 오류를 발생
        if n_items > item_features.shape[0]:
            raise Exception(
                "아이템 피처 행의 수가 아이템 수와 일치하지 않습니다."
            )

        # 사용자 임베딩이 None이 아닌 경우,
        if self.user_embeddings is not None:
            # 사용자 피처 행렬이 추정된 피처 임베딩보다 더 많은 피처를 지정하면 오류를 발생
            if not self.user_embeddings.shape[0] >= user_features.shape[1]:
                raise ValueError(
                    "사용자 피처 행렬이 추정된 피처 임베딩보다 더 많은 피처를 지정합니다: {} vs {}.".format(
                        self.user_embeddings.shape[0], user_features.shape[1]
                    )
                )

        # 아이템 임베딩이 None이 아닌 경우,
        if self.item_embeddings is not None:
            # 아이템 피처 행렬이 추정된 피처 임베딩보다 더 많은 피처를 지정하면 오류를 발생
            if not self.item_embeddings.shape[0] >= item_features.shape[1]:
                raise ValueError(
                    "아이템 피처 행렬이 추정된 피처 임베딩보다 더 많은 피처를 지정합니다: {} vs {}.".format(
                        self.item_embeddings.shape[0], item_features.shape[1]
                    )
                )

        # 사용자 피처와 아이템 피처를 Cython 데이터 타입으로 변환
        user_features = self._to_cython_dtype(user_features)
        item_features = self._to_cython_dtype(item_features)

        # 사용자 피처와 아이템 피처를 반환
        return user_features, item_features

 
    #상호작용 데이터 확인
    def _get_positives_lookup_matrix(self, interactions):

        mat = interactions.tocsr()

        if not mat.has_sorted_indices:
            return mat.sorted_indices()
        else:
            return mat
    #데이터 유형 확인 및 변환
    def _to_cython_dtype(self, mat):

        if mat.dtype != CYTHON_DTYPE:
            return mat.astype(CYTHON_DTYPE)
        else:
            return mat
    #샘플 가중치 처리
    def _process_sample_weight(self, interactions, sample_weight):
        #샘플 가중치가 없지 않는 경우 실행
        if sample_weight is not None:

            if not isinstance(sample_weight, sp.coo_matrix):
                raise ValueError("Sample_weight must be a COO matrix.")

            if sample_weight.shape != interactions.shape:
                raise ValueError(
                    "Sample weight and interactions " "matrices must be the same shape"
                )

            if not (
                np.array_equal(interactions.row, sample_weight.row)
                and np.array_equal(interactions.col, sample_weight.col)
            ):
                raise ValueError(
                    "Sample weight and interaction matrix "
                    "entries must be in the same order"
                )

            if sample_weight.data.dtype != CYTHON_DTYPE:
                sample_weight_data = sample_weight.data.astype(CYTHON_DTYPE)
            else:
                sample_weight_data = sample_weight.data
        else:
            #interatction 데이터가 모두 1.0인지 확인 
            if np.array_equiv(interactions.data, 1.0):
                # # interactions.data를 sample_weight_data에 할당
                sample_weight_data = interactions.data
            else:
                # 새로운 배열 생성
                sample_weight_data = np.ones_like(interactions.data, dtype=CYTHON_DTYPE)

        return sample_weight_data

# LightFM 데이터 반환
    def _get_lightfm_data(self):
        # 필요한 데이터를 딕셔너리 형태로 저장
        lightfm_data = {
            'item_embeddings': self.item_embeddings,
            'item_embedding_gradients': self.item_embedding_gradients,
            'item_embedding_momentum': self.item_embedding_momentum,
            'item_biases': self.item_biases,
            'item_bias_gradients': self.item_bias_gradients,
            'item_bias_momentum': self.item_bias_momentum,
            'user_embeddings': self.user_embeddings,
            'user_embedding_gradients': self.user_embedding_gradients,
            'user_embedding_momentum': self.user_embedding_momentum,
            'user_biases': self.user_biases,
            'user_bias_gradients': self.user_bias_gradients,
            'user_bias_momentum': self.user_bias_momentum,
            'no_components': self.no_components,
            'learning_schedule': int(self.learning_schedule == "adadelta"),
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
        }

        return lightfm_data

    #파라미터가 유한한지 확인
    def _check_finite(self):

        for parameter in (
            self.item_embeddings,
            self.item_biases,
            self.user_embeddings,
            self.user_biases,
        ):
            # A sum of an array that contains non-finite values
            # will also be non-finite, and we avoid creating a
            # large boolean temporary.
            if not np.isfinite(np.sum(parameter)):
                raise ValueError(
                    "Not all estimated parameters are finite,"
                    " your model may have diverged. Try decreasing"
                    " the learning rate or normalising feature values"
                    " and sample weights"
                )
    #입력 데이터가 유한한지 확인
    def _check_input_finite(self, data):

        if not np.isfinite(np.sum(data)):
            raise ValueError(
                "Not all input values are finite. "
                "Check the input for NaNs and infinite values."
            )
    #진행 상황 확인 위한 함수 
    @staticmethod
    def _progress(n, verbose):
        # Use `tqdm` if available,
        # otherwise fallback to `range()`.
        if not verbose:
            return range(n)

        try:
            from tqdm import trange

            return trange(n, desc="Epoch")
        except ImportError:

            def verbose_range():
                for i in range(n):
                    print("Epoch {}".format(i))
                    yield i

            return verbose_range()
    #모델 훈련 함수 
# 모델 학습
    def fit(
        self,
        interactions,
        user_features=None,
        item_features=None,
        sample_weight=None,
        epochs=1,
        num_threads=1,
        verbose=False,
    ):

        self._reset_state()

        interactions = interactions.tocoo()

        if interactions.dtype != CYTHON_DTYPE:
            interactions.data = interactions.data.astype(CYTHON_DTYPE)

        sample_weight_data = self._process_sample_weight(interactions, sample_weight)

        n_users, n_items = interactions.shape
        (user_features, item_features) = self._construct_feature_matrics(
            n_users, n_items, user_features, item_features
        )

        for input_data in (
            user_features.data,
            item_features.data,
            interactions.data,
            sample_weight_data,
        ):
            self._check_input_finite(input_data)
        if self.item_embeddings is None:
            # 잠재변수 초기화
            self._initialize(
                self.no_components, item_features.shape[1], user_features.shape[1]
            )
        
        if not item_features.shape[1] == self.item_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in item_features")

        if not user_features.shape[1] == self.user_embeddings.shape[0]:
            raise ValueError("Incorrect number of features in user_features")

        if num_threads < 1:
            raise ValueError("Number of threads must be 1 or larger.")

        for _ in self._progress(epochs, verbose=verbose):
            self._run_epoch(
                item_features,
                user_features,
                interactions,
                sample_weight_data,
                num_threads
            )

        self._check_finite()

        return self

    #에폭 실행 함수 
    def _run_epoch(
        self,
        item_features,
        user_features,
        interactions,
        sample_weight,
        num_threads
    ):
        
        positives_lookup = CSRMatrix(
            self._get_positives_lookup_matrix(interactions)
        )

        # shuffle index 생성 
        shuffle_indices = np.arange(len(interactions.data), dtype=np.int32)
        self.random_state.shuffle(shuffle_indices)

        lightfm_data = self._get_lightfm_data()

#헤당 코드의 함수 구현 불가 -> _lightfm_fast.pyx 파일 참고
    def predict(self, user_ids, item_ids, item_features=None, user_features=None, num_threads=1):
        predictions = predict_lightfm(
            CSRMatrix(user_features),
            CSRMatrix(item_features),
            user_ids,
            item_ids,
        )
        return predictions

