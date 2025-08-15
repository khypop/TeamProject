// React와 React Native의 핵심 라이브러리들을 가져옴
import React, {useState} from 'react';
// React Native의 기본 UI 컴포넌트들을 가져옴
import {
  View,           // 컨테이너 역할을 하는 기본 뷰 컴포넌트
  Text,            // 텍스트를 표시하는 컴포넌트
  TouchableOpacity, // 터치 가능한 투명도 조절 버튼
  Image,           // 이미지를 표시하는 컴포넌트
  StyleSheet,      // 스타일을 정의하는 유틸리티
  ScrollView,      // 스크롤 가능한 뷰
  Alert,           // 알림창을 표시하는 컴포넌트
  TextInput,       // 텍스트 입력 필드
  ActivityIndicator, // 로딩 스피너
} from 'react-native';
// 이미지 선택을 위한 라이브러리 (갤러리에서 이미지 선택)
import {launchImageLibrary} from 'react-native-image-picker';
// 파일 시스템 접근을 위한 라이브러리 (Base64 변환용)
import RNFS from 'react-native-fs';
// 백엔드 API와 통신하기 위한 이미지 분류 함수
import {classifyImage} from '../services/api';

/**
 * ImageClassifier 컴포넌트
 * CLIP 모델을 사용하여 이미지를 분류하는 메인 컴포넌트
 */
const ImageClassifier = () => {
  // 선택된 이미지 정보를 저장하는 상태 (null = 이미지 미선택)
  const [selectedImage, setSelectedImage] = useState(null);
  
  // 분류할 카테고리 목록을 저장하는 상태 (기본값: 6개 카테고리)
  const [categories, setCategories] = useState([
    '동물',    // 동물 관련 이미지
    '음식',    // 음식 관련 이미지
    '자연',    // 자연 풍경
    '사람',    // 사람 관련 이미지
    '건물',    // 건축물
    '차량'     // 자동차, 기차 등
  ]);
  
  // 사용자가 새로 추가하려는 카테고리 텍스트를 저장하는 상태
  const [customCategory, setCustomCategory] = useState('');
  
  // 이미지 분류 결과를 저장하는 상태 (배열 형태)
  const [results, setResults] = useState([]);
  
  // API 호출 중인지 여부를 나타내는 로딩 상태
  const [loading, setLoading] = useState(false);

  /**
   * 이미지 선택 함수
   * 사용자가 갤러리에서 이미지를 선택할 수 있게 함
   */
  const selectImage = () => {
    // 이미지 선택 옵션 설정
    const options = {
      mediaType: 'photo',        // 사진만 선택 가능 (비디오 제외)
      quality: 0.8,              // 이미지 품질 (0.8 = 80%)
      maxWidth: 1024,            // 최대 너비 제한
      maxHeight: 1024,           // 최대 높이 제한
    };

    // 이미지 라이브러리 실행
    launchImageLibrary(options, response => {
      // 사용자가 취소하거나 오류가 발생한 경우
      if (response.didCancel || response.error) {
        return; // 함수 종료
      }

      // 이미지가 성공적으로 선택된 경우
      if (response.assets && response.assets[0]) {
        setSelectedImage(response.assets[0]);  // 선택된 이미지 저장
        setResults([]);                        // 이전 분류 결과 초기화
      }
    });
  };

  /**
   * 새로운 카테고리 추가 함수
   * 사용자가 입력한 텍스트를 카테고리 목록에 추가
   */
  const addCategory = () => {
    // 입력된 텍스트가 비어있지 않고, 중복되지 않는 경우에만 추가
    if (customCategory.trim() && !categories.includes(customCategory.trim())) {
      setCategories([...categories, customCategory.trim()]); // 새 카테고리 추가
      setCustomCategory(''); // 입력 필드 초기화
    }
  };

  /**
   * 카테고리 제거 함수
   * @param {number} index - 제거할 카테고리의 인덱스
   */
  const removeCategory = (index) => {
    // 해당 인덱스를 제외한 새로운 카테고리 배열 생성
    const newCategories = categories.filter((_, i) => i !== index);
    setCategories(newCategories); // 상태 업데이트
  };

  /**
   * 이미지를 Base64 문자열로 변환하는 함수
   * @param {string} imageUri - 변환할 이미지의 URI
   * @returns {Promise<string>} Base64로 인코딩된 이미지 문자열
   */
  const convertImageToBase64 = async (imageUri) => {
    try {
      // RNFS를 사용하여 이미지 파일을 Base64로 읽기
      const base64 = await RNFS.readFile(imageUri, 'base64');
      return base64;
    } catch (error) {
      console.error('Base64 변환 오류:', error);
      throw error; // 오류를 상위로 전파
    }
  };

  /**
   * 이미지 분류를 실행하는 메인 함수
   * 선택된 이미지를 백엔드로 전송하여 분류 결과를 받아옴
   */
  const performClassification = async () => {
    // 이미지가 선택되지 않은 경우 오류 알림
    if (!selectedImage) {
      Alert.alert('오류', '이미지를 먼저 선택해주세요.');
      return;
    }

    // 카테고리가 하나도 없는 경우 오류 알림
    if (categories.length === 0) {
      Alert.alert('오류', '최소한 하나의 카테고리를 추가해주세요.');
      return;
    }

    // 로딩 상태 시작 및 결과 초기화
    setLoading(true);
    setResults([]);

    try {
      // 이미지를 Base64로 변환
      const base64Image = await convertImageToBase64(selectedImage.uri);
      
      // 백엔드 API 호출하여 이미지 분류 실행
      const result = await classifyImage(base64Image, categories);

      // 분류 성공 시 결과 저장
      if (result.success) {
        setResults(result.results);
      } else {
        // 분류 실패 시 오류 메시지 표시
        Alert.alert('분류 실패', result.error || '알 수 없는 오류가 발생했습니다.');
      }
    } catch (error) {
      // 네트워크 오류 등 예외 상황 처리
      console.error('분류 오류:', error);
      Alert.alert('오류', '서버와 통신 중 오류가 발생했습니다.');
    } finally {
      // 로딩 상태 종료 (성공/실패와 관계없이)
      setLoading(false);
    }
  };

  /**
   * 분류 결과를 렌더링하는 함수
   * @param {Object} result - 분류 결과 객체 (category, confidence, percentage 포함)
   * @param {number} index - 결과 배열의 인덱스
   * @returns {JSX.Element} 결과 항목을 표시하는 UI 컴포넌트
   */
  const renderResult = (result, index) => (
    <View key={index} style={styles.resultItem}>
      {/* 카테고리 이름 표시 */}
      <Text style={styles.categoryText}>{result.category}</Text>
      
      {/* 신뢰도 바와 퍼센트 표시 */}
      <View style={styles.confidenceContainer}>
        {/* 신뢰도에 따라 너비가 변하는 파란색 바 */}
        <View 
          style={[
            styles.confidenceBar, 
            {width: `${result.confidence * 100}%`} // confidence를 퍼센트로 변환
          ]} 
        />
        {/* 신뢰도 퍼센트 텍스트 */}
        <Text style={styles.confidenceText}>{result.percentage}</Text>
      </View>
    </View>
  );

  // 메인 UI 렌더링
  return (
    <ScrollView style={styles.container}>
      {/* 앱 제목 */}
      <Text style={styles.title}>CLIP 이미지 분류기</Text>

      {/* 이미지 선택 영역 */}
      <TouchableOpacity style={styles.imageContainer} onPress={selectImage}>
        {selectedImage ? (
          // 이미지가 선택된 경우: 선택된 이미지 표시
          <Image source={{uri: selectedImage.uri}} style={styles.image} />
        ) : (
          // 이미지가 선택되지 않은 경우: 플레이스홀더 표시
          <View style={styles.placeholderContainer}>
            <Text style={styles.placeholderText}>이미지 선택</Text>
          </View>
        )}
      </TouchableOpacity>

      {/* 카테고리 관리 섹션 */}
      <View style={styles.categorySection}>
        <Text style={styles.sectionTitle}>분류 카테고리</Text>
        
        {/* 새 카테고리 추가 입력 필드와 버튼 */}
        <View style={styles.addCategoryContainer}>
          <TextInput
            style={styles.categoryInput}
            placeholder="새 카테고리 추가"
            value={customCategory}
            onChangeText={setCustomCategory}
            onSubmitEditing={addCategory} // 엔터키로도 추가 가능
          />
          <TouchableOpacity style={styles.addButton} onPress={addCategory}>
            <Text style={styles.addButtonText}>추가</Text>
          </TouchableOpacity>
        </View>

        {/* 기존 카테고리 목록 (칩 형태로 표시) */}
        <View style={styles.categoriesList}>
          {categories.map((category, index) => (
            <TouchableOpacity
              key={index}
              style={styles.categoryChip}
              onLongPress={() => removeCategory(index)} // 길게 누르면 삭제
            >
              <Text style={styles.categoryChipText}>{category}</Text>
            </TouchableOpacity>
          ))}
        </View>
        {/* 사용법 안내 텍스트 */}
        <Text style={styles.helpText}>카테고리를 길게 눌러서 삭제</Text>
      </View>

      {/* 이미지 분류 실행 버튼 */}
      <TouchableOpacity 
        style={[styles.classifyButton, loading && styles.disabledButton]} // 로딩 중일 때 비활성화 스타일 적용
        onPress={performClassification}
        disabled={loading} // 로딩 중일 때 버튼 비활성화
      >
        {loading ? (
          // 로딩 중일 때: 스피너 표시
          <ActivityIndicator color="#fff" />
        ) : (
          // 로딩 중이 아닐 때: 버튼 텍스트 표시
          <Text style={styles.classifyButtonText}>이미지 분류 실행</Text>
        )}
      </TouchableOpacity>

      {/* 분류 결과 표시 섹션 (결과가 있을 때만 표시) */}
      {results.length > 0 && (
        <View style={styles.resultsSection}>
          <Text style={styles.sectionTitle}>분류 결과</Text>
          {/* 각 결과 항목을 renderResult 함수로 렌더링 */}
          {results.map(renderResult)}
        </View>
      )}
    </ScrollView>
  );
};

// 컴포넌트의 스타일 정의
const styles = StyleSheet.create({
  // 메인 컨테이너 스타일
  container: {
    flex: 1,                    // 전체 화면 차지
    backgroundColor: '#f5f5f5', // 연한 회색 배경
    padding: 20,                // 모든 방향에 20px 패딩
  },
  
  // 앱 제목 스타일
  title: {
    fontSize: 24,               // 글자 크기 24px
    fontWeight: 'bold',         // 굵은 글씨
    textAlign: 'center',        // 가운데 정렬
    marginBottom: 20,           // 아래쪽 여백 20px
    color: '#333',              // 진한 회색 글씨
  },
  
  // 이미지 컨테이너 스타일
  imageContainer: {
    height: 200,                // 고정 높이 200px
    backgroundColor: '#fff',     // 흰색 배경
    borderRadius: 10,           // 둥근 모서리 10px
    marginBottom: 20,           // 아래쪽 여백 20px
    overflow: 'hidden',         // 내용이 넘치면 숨김
    elevation: 3,               // Android 그림자 효과
    shadowColor: '#000',        // iOS 그림자 색상
    shadowOffset: {width: 0, height: 2}, // iOS 그림자 위치
    shadowOpacity: 0.1,         // iOS 그림자 투명도
    shadowRadius: 4,            // iOS 그림자 블러 효과
  },
  
  // 선택된 이미지 스타일
  image: {
    width: '100%',              // 컨테이너 너비의 100%
    height: '100%',             // 컨테이너 높이의 100%
    resizeMode: 'cover',        // 이미지가 컨테이너를 꽉 채우도록 조정
  },
  
  // 이미지 미선택 시 플레이스홀더 컨테이너 스타일
  placeholderContainer: {
    flex: 1,                    // 남은 공간 모두 차지
    justifyContent: 'center',   // 세로 중앙 정렬
    alignItems: 'center',       // 가로 중앙 정렬
  },
  
  // 플레이스홀더 텍스트 스타일
  placeholderText: {
    fontSize: 16,               // 글자 크기 16px
    color: '#666',              // 중간 회색 글씨
  },
  
  // 카테고리 섹션 스타일
  categorySection: {
    backgroundColor: '#fff',     // 흰색 배경
    borderRadius: 10,           // 둥근 모서리 10px
    padding: 15,                // 내부 여백 15px
    marginBottom: 20,           // 아래쪽 여백 20px
    elevation: 3,               // Android 그림자 효과
    shadowColor: '#000',        // iOS 그림자 색상
    shadowOffset: {width: 0, height: 2}, // iOS 그림자 위치
    shadowOpacity: 0.1,         // iOS 그림자 투명도
    shadowRadius: 4,            // iOS 그림자 블러 효과
  },
  
  // 섹션 제목 스타일
  sectionTitle: {
    fontSize: 18,               // 글자 크기 18px
    fontWeight: 'bold',         // 굵은 글씨
    marginBottom: 15,           // 아래쪽 여백 15px
    color: '#333',              // 진한 회색 글씨
  },
  
  // 카테고리 추가 컨테이너 스타일 (입력 필드와 버튼을 가로로 배치)
  addCategoryContainer: {
    flexDirection: 'row',       // 가로 방향 배치
    marginBottom: 15,           // 아래쪽 여백 15px
  },
  
  // 카테고리 입력 필드 스타일
  categoryInput: {
    flex: 1,                    // 남은 공간 모두 차지
    borderWidth: 1,             // 테두리 두께 1px
    borderColor: '#ddd',        // 연한 회색 테두리
    borderRadius: 5,            // 둥근 모서리 5px
    paddingHorizontal: 10,      // 좌우 내부 여백 10px
    paddingVertical: 8,         // 상하 내부 여백 8px
    marginRight: 10,            // 오른쪽 여백 10px
  },
  
  // 카테고리 추가 버튼 스타일
  addButton: {
    backgroundColor: '#007bff',  // 파란색 배경
    borderRadius: 5,            // 둥근 모서리 5px
    paddingHorizontal: 15,      // 좌우 내부 여백 15px
    paddingVertical: 8,         // 상하 내부 여백 8px
    justifyContent: 'center',   // 내용 세로 중앙 정렬
  },
  
  // 카테고리 추가 버튼 텍스트 스타일
  addButtonText: {
    color: '#fff',              // 흰색 글씨
    fontWeight: 'bold',         // 굵은 글씨
  },
  
  // 카테고리 목록 컨테이너 스타일
  categoriesList: {
    flexDirection: 'row',       // 가로 방향 배치
    flexWrap: 'wrap',           // 줄바꿈 허용
    marginBottom: 10,           // 아래쪽 여백 10px
  },
  
  // 개별 카테고리 칩 스타일
  categoryChip: {
    backgroundColor: '#e9ecef',  // 연한 회색 배경
    borderRadius: 20,           // 둥근 모서리 20px (원형에 가까움)
    paddingHorizontal: 15,      // 좌우 내부 여백 15px
    paddingVertical: 8,         // 상하 내부 여백 8px
    margin: 4,                  // 모든 방향에 4px 여백
  },
  
  // 카테고리 칩 텍스트 스타일
  categoryChipText: {
    color: '#495057',           // 중간 회색 글씨
    fontSize: 14,               // 글자 크기 14px
  },
  
  // 도움말 텍스트 스타일
  helpText: {
    fontSize: 12,               // 글자 크기 12px
    color: '#6c757d',           // 연한 회색 글씨
    fontStyle: 'italic',        // 기울임꼴
  },
  
  // 분류 실행 버튼 스타일
  classifyButton: {
    backgroundColor: '#28a745',  // 초록색 배경
    borderRadius: 10,           // 둥근 모서리 10px
    paddingVertical: 15,        // 상하 내부 여백 15px
    alignItems: 'center',       // 내용 가로 중앙 정렬
    marginBottom: 20,           // 아래쪽 여백 20px
    elevation: 3,               // Android 그림자 효과
    shadowColor: '#000',        // iOS 그림자 색상
    shadowOffset: {width: 0, height: 2}, // iOS 그림자 위치
    shadowOpacity: 0.1,         // iOS 그림자 투명도
    shadowRadius: 4,            // iOS 그림자 블러 효과
  },
  
  // 비활성화된 버튼 스타일 (로딩 중일 때)
  disabledButton: {
    backgroundColor: '#6c757d',  // 회색 배경
  },
  
  // 분류 실행 버튼 텍스트 스타일
  classifyButtonText: {
    color: '#fff',              // 흰색 글씨
    fontSize: 16,               // 글자 크기 16px
    fontWeight: 'bold',         // 굵은 글씨
  },
  
  // 결과 표시 섹션 스타일
  resultsSection: {
    backgroundColor: '#fff',     // 흰색 배경
    borderRadius: 10,           // 둥근 모서리 10px
    padding: 15,                // 내부 여백 15px
    marginBottom: 20,           // 아래쪽 여백 20px
    elevation: 3,               // Android 그림자 효과
    shadowColor: '#000',        // iOS 그림자 색상
    shadowOffset: {width: 0, height: 2}, // iOS 그림자 위치
    shadowOpacity: 0.1,         // iOS 그림자 투명도
    shadowRadius: 4,            // iOS 그림자 블러 효과
  },
  
  // 개별 결과 항목 스타일
  resultItem: {
    marginBottom: 15,           // 아래쪽 여백 15px
  },
  
  // 결과 카테고리 텍스트 스타일
  categoryText: {
    fontSize: 16,               // 글자 크기 16px
    fontWeight: 'bold',         // 굵은 글씨
    marginBottom: 5,            // 아래쪽 여백 5px
    color: '#333',              // 진한 회색 글씨
  },
  
  // 신뢰도 표시 컨테이너 스타일 (바와 텍스트를 가로로 배치)
  confidenceContainer: {
    flexDirection: 'row',       // 가로 방향 배치
    alignItems: 'center',       // 세로 중앙 정렬
  },
  
  // 신뢰도 바 스타일
  confidenceBar: {
    height: 8,                  // 높이 8px
    backgroundColor: '#007bff',  // 파란색 배경
    borderRadius: 4,            // 둥근 모서리 4px
    marginRight: 10,            // 오른쪽 여백 10px
    minWidth: 20,               // 최소 너비 20px (아주 낮은 신뢰도도 표시)
  },
  
  // 신뢰도 퍼센트 텍스트 스타일
  confidenceText: {
    fontSize: 14,               // 글자 크기 14px
    color: '#666',              // 중간 회색 글씨
    fontWeight: '600',          // 중간 굵기 글씨
  },
});

// 컴포넌트를 다른 파일에서 사용할 수 있도록 내보내기
export default ImageClassifier;
