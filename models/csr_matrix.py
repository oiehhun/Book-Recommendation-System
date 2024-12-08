import numpy as np  # numpy 라이브러리
import scipy.sparse as sp  # scipy의 sparse 모듈

class CSRMatrix:  # CSRMatrix라는 클래스 정의
    def __init__(self, matrix):  # 생성자 함수 정의, 이 함수는 객체가 생성될 때 호출
        self.data = matrix.data  # 데이터 배열을 저장
        self.indices = matrix.indices  # 인덱스 배열 저장
        self.indptr = matrix.indptr  # 인덱스 포인터 배열 저장
        self.shape = matrix.shape  # 행렬의 형태를 저장
        self.has_sorted_indices = True  # 인덱스가 정렬되었는지 여부 저장

    def row(self, idx):  # 특정 행의 데이터를 반환하는 메소드 정의
        return self.data[self.indptr[idx] : self.indptr[idx + 1]]  # 행의 시작과 끝을 나타내는 인덱스 사용하여 데이터를 슬라이싱

    def sorted_indices(self):  # 정렬된 인덱스를 반환하는 메소드 정의
        sorted_indices = np.lexsort(
            (self.indices, self.indptr)
        )  # 행과 열 인덱스를 기준으로 인덱스를 정렬
        return CSRMatrix(
            sp.csr_matrix(
                (self.data[sorted_indices], self.indices[sorted_indices], self.indptr),  # 정렬된 인덱스를 사용하여 새로운 CSR 행렬을 생성합니다.
                shape=self.shape,  # 행렬의 형태를 지정
            )
        )
