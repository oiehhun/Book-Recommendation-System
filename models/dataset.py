import array
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing

class _IncrementalCOOMatrix(object):
    #클래스 초기화 함수 
    def __init__(self, shape, dtype):
        #데이터 타입에 따라 type_flag 지정 
        if dtype is np.int32:
            type_flag = "i"
        elif dtype is np.int64:
            type_flag = "l"
        elif dtype is np.float32:
            type_flag = "f"
        elif dtype is np.float64:
            type_flag = "d"
        else:
        #지원되지 않는 데이터 타입의 경우 오류 발생 
            raise Exception("Dtype 데이터 타입 오류")
        #shape 저장 
        self.shape = shape
        #dtype 저장 
        self.dtype = dtype
        #행, 열, 데이터를 저장할 배열 초기화 
        self.rows = array.array("i")
        self.cols = array.array("i")
        self.data = array.array(type_flag)
    #데이터 추가 함수 
    def append(self, i, j, v):
        m, n = self.shape
        #인덱스 벗어날 경우 오류 발생 
        if i >= m or j >= n:
            raise Exception("인데스 오류 ")
        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)
    #coo_matrix로 변환하는 함수
    def tocoo(self):
        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)
        #coco 행령 생성 및 반환 
        return sp.coo_matrix((data, (rows, cols)), shape=self.shape)
    #데이터의 길이 반환 함수 
    def __len__(self):
        #데이터의 길이 반환 
        return len(self.data)

#FeatureBuilder 클래스 생성 
class _FeatureBuilder(object):
    #초기화 함수 
    def __init__(
        self, id_mapping, feature_mapping, identity_features, normalize, entity_type
    ):
    #id mapping, feature mapping, identity_features, normalize, entity_type 멤버 변수 설정 
        self._id_mapping = id_mapping
        self._feature_mapping = feature_mapping
        self._identity_features = identity_features
        self._normalize = normalize
        self._entity_type = entity_type
    #특성의 형태 반환 함수 
    def features_shape(self):
        #id mapping, feature mapping의 길이 반환
        return len(self._id_mapping), len(self._feature_mapping)
    #특성을 반복 순회하는 함수
    def _iter_features(self, features):
        #특성이 딕셔너리인 경우 
        if isinstance(features, dict):
            for entry in features.items():
                yield entry
        #특성이 딕셔너리 형태가 아닌 경우 특성명과 함께 1.0 반환
        else:
            for feature_name in features:
                yield (feature_name, 1.0)
    #특성을 처리하는 함수
    def _process_features(self, datum):
        #특성의 길이가 2가 아닌 경우 오류 발생
        if len(datum) != 2:
            raise ValueError(
                "Expected tuples of ({}_id, features), "
                "got {}.".format(self._entity_type, datum)
            )
        #entity_id, features 변수에 데이터 저장
        entity_id, features = datum
        #entity_id가 id_mapping에 없는 경우 오류 발생
        if entity_id not in self._id_mapping:
            raise ValueError(
                "{entity_type} id {entity_id} not in {entity_type} id mappings.".format(
                    entity_type=self._entity_type, entity_id=entity_id
                )
            )
        #entity_id의 인덱스 저장
        idx = self._id_mapping[entity_id]
        #특성을 순회하며 특성의 인덱스와 가중치 반환
        for (feature, weight) in self._iter_features(features):
            if feature not in self._feature_mapping:
                raise ValueError(
                    "Feature {} not in feature mapping. "
                    "Call fit first.".format(feature)
                )
            #특성의 인덱스 저장
            feature_idx = self._feature_mapping[feature]
            #특성의 인덱스와 가중치 반환
            yield (idx, feature_idx, weight)
    #데이터를 기반으로 특성을 구축하는 함수
    def build(self, data):
        #IncrementalCOOMatrix 클래스의 객체 생성
        features = _IncrementalCOOMatrix(self.features_shape(), np.float32)
        #
        if self._identity_features:
            for (_id, idx) in self._id_mapping.items():
                features.append(idx, self._feature_mapping[_id], 1.0)
        #데이터를 순회하며 특성을 추가
        for datum in data:
            for (entity_idx, feature_idx, weight) in self._process_features(datum):
                features.append(entity_idx, feature_idx, weight)

        features = features.tocoo().tocsr()
        #normalize가 True인 경우
        if self._normalize:
            if np.any(features.getnnz(1) == 0):
                raise ValueError(
                    "Cannot normalize feature matrix: some rows have zero norm. "
                    "Ensure that features were provided for all entries."
                )

            sklearn.preprocessing.normalize(features, norm="l1", copy=False)
        #특성 반환
        return features


class Dataset(object):
#유저와 아이템간의 상호작용 행렬을 생성하는 클래스 
    #클래스 초기화 함수 
    def __init__(self, user_identity_features=True, item_identity_features=True):
        #사용자와 아이템에 대한 고유 특성 
        self._user_identity_features = user_identity_features
        self._item_identity_features = item_identity_features
        #사용자와 아이템에 대한 id mapping, feature mapping 초기화
        self._user_id_mapping = {}
        self._item_id_mapping = {}
        self._user_feature_mapping = {}
        self._item_feature_mapping = {}
    #데이터를 기반으로 유저 및 아이템 맵핑을 생성하는 함수
    def fit(self, users, items, user_features=None, item_features=None):
        self._user_id_mapping = {}
        self._item_id_mapping = {}
        self._user_feature_mapping = {}
        self._item_feature_mapping = {}
        return self.fit_partial(users, items, user_features, item_features)

    def fit_partial(
        self, users=None, items=None, user_features=None, item_features=None
    ):
        if users is not None:
            for user_id in users:
                self._user_id_mapping.setdefault(user_id, len(self._user_id_mapping))

                if self._user_identity_features:
                    self._user_feature_mapping.setdefault(
                        user_id, len(self._user_feature_mapping)
                    )

        if items is not None:
            for item_id in items:
                self._item_id_mapping.setdefault(item_id, len(self._item_id_mapping))

                if self._item_identity_features:
                    self._item_feature_mapping.setdefault(
                        item_id, len(self._item_feature_mapping)
                    )
        if user_features is not None:
            for user_feature in user_features:
                self._user_feature_mapping.setdefault(
                    user_feature, len(self._user_feature_mapping)
                )

        if item_features is not None:
            for item_feature in item_features:
                self._item_feature_mapping.setdefault(
                    item_feature, len(self._item_feature_mapping)
                )
    #데이터 분해 함수 
    def _unpack_datum(self, datum):
        #데이터의 길이가 3인 경우 
        if len(datum) == 3:
            #사용자 id, 아이템 id, 가중치 분해 
            (user_id, item_id, weight) = datum
        #데이터의 길이가 2인 경우
        elif len(datum) == 2:
            #사용자 id, 아이템 id 분해하고 가중치 1.0으로 설정 
            (user_id, item_id) = datum
            weight = 1.0
        else:
            #그 외의 경우 오류 발생 
            raise ValueError(
                "Expecting tuples of (user_id, item_id, weight) "
                "or (user_id, item_id). Got {}".format(datum)
            )
        #사용자 id, 아이템 id를 기반으로 인덱스 반환
        user_idx = self._user_id_mapping.get(user_id)
        item_idx = self._item_id_mapping.get(item_id)
        #사용자 id가 id mapping에 없는 경우 오류 발생
        if user_idx is None:
            #사용자 id가 id mapping에 없는 경우 오류 발생
            raise ValueError(
                "User id {} not in user id mapping. Make sure "
                "you call the fit method.".format(user_id)
            )
        #아이템 id가 id mapping에 없는 경우 오류 발생
        if item_idx is None:
            #아이템 id가 id mapping에 없는 경우 오류 발생
            raise ValueError(
                "Item id {} not in item id mapping. Make sure "
                "you call the fit method.".format(item_id)
            )
        #사용자 인덱스,ㅇ 아이템 인덱스, 가중치 반환
        return (user_idx, item_idx, weight)
    #상호작용 행렬의 형태 반환 함수
    def interactions_shape(self):
        #사용자 id mapping, 아이템 id mapping의 길이 반환
        return (len(self._user_id_mapping), len(self._item_id_mapping))
    #상호작용 행렬을 생성하는 함수 
    def build_interactions(self, data):
        #IncrementalCOOMatrix 클래스의 객체 생성
        interactions = _IncrementalCOOMatrix(self.interactions_shape(), np.int32)
        weights = _IncrementalCOOMatrix(self.interactions_shape(), np.float32)
        #데이터를 순회하며 상호작용 행렬에 추가
        for datum in data:
            #데이터를 분해하여 유저 인덱스, 아이템 인덱스, 가중치를 가져옴 
            user_idx, item_idx, weight = self._unpack_datum(datum)
            #상호작용 행렬에 데이터 추가
            interactions.append(user_idx, item_idx, 1)
            weights.append(user_idx, item_idx, weight)
        #상호작용 행렬 반환
        return (interactions.tocoo(), weights.tocoo())
    #사용자 특성의 형태 반환 함수
    def user_features_shape(self):
        #사용자 id mapping, 사용자 feature mapping의 길이 반환
        return (len(self._user_id_mapping), len(self._user_feature_mapping))
    #사용자 특성을 구축하는 함수 
    def build_user_features(self, data, normalize=True):
        #builder 객체 생성 
        builder = _FeatureBuilder(
            self._user_id_mapping,
            self._user_feature_mapping,
            self._user_identity_features,
            normalize,
            "user",
        )
        #사용자 특성 반환
        return builder.build(data)
    #아이템 특성의 형태 반환 함수
    def item_features_shape(self):
        #아이템 id mapping, 아이템 feature mapping의 길이 반환
        return (len(self._item_id_mapping), len(self._item_feature_mapping))
    #아이템 특성을 구축하는 함수
    def build_item_features(self, data, normalize=True):
        #builder 객체 생성
        builder = _FeatureBuilder(
            self._item_id_mapping,
            self._item_feature_mapping,
            self._item_identity_features,
            normalize,
            "item",
        )

        return builder.build(data)
    #모델의 차원 반환 함수
    def model_dimensions(self):
        #사용자 feature mapping, 아이템 feature mapping의 길이 반환
        return (len(self._user_feature_mapping), len(self._item_feature_mapping))
    #매핑 반환 함수
    def mapping(self):
        #사용자 id mapping, 사용자 feature mapping, 아이템 id mapping, 아이템 feature mapping 반환
        return (
            self._user_id_mapping,
            self._user_feature_mapping,
            self._item_id_mapping,
            self._item_feature_mapping,
        )