import React, { useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Button,
  FlatList,
  StyleSheet,
  Text,
  TextInput,
  View,
  Platform,
  Linking,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import * as FileSystem from "expo-file-system"; // SAF + 파일 복사
import * as IntentLauncher from "expo-intent-launcher"; // 설정 화면 열기
import * as Application from "expo-application"; // 패키지명
import Slider from "@react-native-community/slider";
import axios from "axios";

// ⚠️ 에뮬레이터: 10.0.2.2 / 실기기: PC 로컬 IP
// 예) const BASE_URL = "http://192.168.0.23:8000";
const BASE_URL = "http://10.0.2.2:8000";
const SAF = FileSystem.StorageAccessFramework;

// content:// → file://로 복사 (서버가 읽을 수 있게)
const toUploadable = async (f) => {
  if (f?.uri?.startsWith("content://")) {
    const safe = (f.name || "image.jpg").replace(/[^a-zA-Z0-9._-]/g, "_");
    const dest = `${FileSystem.cacheDirectory}up-${Date.now()}-${safe}`;
    await FileSystem.copyAsync({ from: f.uri, to: dest });
    return { ...f, uri: dest };
  }
  return f;
};

export default function App() {
  const [files, setFiles] = useState([]); // { uri, name, mimeType }
  const [prompt, setPrompt] = useState("");
  const [threshold, setThreshold] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ cur: 0, total: 1 });
  const [results, setResults] = useState([]); // [{ name, score }]
  const [hasSearched, setHasSearched] = useState(false);
  const [dirUri, setDirUri] = useState(null);

  const BATCH = 8; // 느리면 16

  /**
   * (선택) Android의 "모든 파일 접근" 설정 화면으로 이동
   * - MANAGE_EXTERNAL_STORAGE 토글 화면 (Android 11+)
   * - Expo Go에서는 권한을 붙일 수 없으니 Dev Client에서 사용
   */
  const openAllFilesAccessSettings = async () => {
    if (Platform.OS !== "android" || Platform.Version < 30) {
      Alert.alert("안내", "이 기능은 Android 11+에서만 필요합니다.");
      return;
    }
    try {
      const pkg = Application.applicationId;
      await IntentLauncher.startActivityAsync(
        // 앱별 모든 파일 접근 권한 화면
        "android.settings.MANAGE_APP_ALL_FILES_ACCESS_PERMISSION",
        { data: `package:${pkg}` }
      );
    } catch (e1) {
      try {
        // 전체 목록 화면
        await IntentLauncher.startActivityAsync(
          "android.settings.MANAGE_ALL_FILES_ACCESS_PERMISSION"
        );
      } catch (e2) {
        try {
          await Linking.openSettings();
        } catch { }
      }
    }
  };

  /**
   * SAF로 디렉터리 선택 → 내부 이미지 수집
   * 일부 상위 루트(예: Internal Storage 루트, Android/data)는 보안상 막혀 "Can't use this folder"가 뜸.
   * → Pictures/DCIM/Download 같은 하위 폴더로 들어가 "이 폴더 사용"을 누르세요.
   */
  const pickFolderAndLoad = async () => {
    try {
      if (Platform.OS !== "android") {
        Alert.alert("안내", "폴더 선택은 Android에서만 지원됩니다.");
        return;
      }

      // 초기 위치를 내부 저장소 루트로 지정 (선택 UX 개선)
      const perm = await SAF.requestDirectoryPermissionsAsync(
        "content://com.android.externalstorage.documents/root/primary"
      );

      if (!perm.granted) {
        Alert.alert(
          "권한 필요",
          "선택한 폴더 접근 권한을 허용해주세요. 상위 루트가 차단되면 Pictures/DCIM/Download 같은 하위 폴더를 선택하세요."
        );
        return;
      }

      const directoryUri = perm.directoryUri;
      setDirUri(directoryUri);

      let uris;
      try {
        uris = await SAF.readDirectoryAsync(directoryUri);
      } catch (err) {
        Alert.alert(
          "접근 불가",
          "이 폴더는 보안상 접근이 제한됩니다. 내부의 하위 폴더(Pictures, DCIM 등)로 이동해 다시 시도하거나, 모든 파일 접근 권한을 활성화하세요."
        );
        return;
      }

      const imageUris = uris.filter((u) => /\.(jpe?g|png|webp|gif|bmp)$/i.test(decodeURIComponent(u)));
      const items = imageUris.map((u) => {
        const decoded = decodeURIComponent(u);
        let base = decoded.split("/").pop() || "image";
        base = base.replace(/^document\//, "");
        const lower = base.toLowerCase();
        const type = lower.endsWith(".png")
          ? "image/png"
          : lower.endsWith(".webp")
            ? "image/webp"
            : lower.endsWith(".gif")
              ? "image/gif"
              : lower.endsWith(".bmp")
                ? "image/bmp"
                : "image/jpeg";
        return { uri: u, name: base, mimeType: type };
      });

      setFiles(items);
      setResults([]);
      setHasSearched(false);
      Alert.alert("완료", `선택한 폴더에서 이미지 ${items.length}장을 불러왔습니다.`);
    } catch (e) {
      Alert.alert("오류", String(e?.message || e));
    }
  };

  /** 파일 앱에서 수동 다중 선택 (백업 경로) */
  const pickImages = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: "image/*",
        multiple: true,
        copyToCacheDirectory: false,
      });
      if (res.canceled) return;
      const items = (res.assets || []).map((a) => ({
        uri: a.uri,
        name: a.name || "image.jpg",
        mimeType: a.mimeType || "image/jpeg",
      }));
      setFiles(items);
      setResults([]);
      setHasSearched(false);
    } catch (e) {
      Alert.alert("선택 오류", String(e?.message || e));
    }
  };

  const clearSelection = () => {
    setFiles([]);
    setResults([]);
    setHasSearched(false);
  };

  /** 서버 검색 (배치 업로드) */
  const search = async () => {
    if (!files.length) {
      Alert.alert("안내", "폴더를 선택하거나 이미지를 선택하세요.");
      return;
    }
    if (!prompt.trim()) {
      Alert.alert("안내", "텍스트 프롬프트를 입력하세요.");
      return;
    }

    setLoading(true);
    setResults([]);
    setHasSearched(false);
    setProgress({ cur: 0, total: files.length });

    // 서버 연결 확인 (네트워크 에러 구분)
    try { await axios.get(`${BASE_URL}/health`, { timeout: 3000 }); }
    catch { Alert.alert("서버 연결 실패", "BASE_URL, 방화벽, 서버 실행 상태를 확인하세요."); setLoading(false); return; }

    try {
      let all = [];
      for (let i = 0; i < files.length; i += BATCH) {
        const chunk = files.slice(i, i + BATCH);
        const prepared = await Promise.all(chunk.map(toUploadable));
        const form = new FormData();
        form.append("text", prompt.trim());
        form.append("threshold", String(threshold));
        for (const f of prepared) {
          form.append("files", {
            uri: f.uri, // 우선 content:// 그대로 업로드 시도
            name: f.name || "image.jpg",
            type: f.mimeType || "image/jpeg",
          });
        }

        const { data } = await axios.post(`${BASE_URL}/search`, form, {
          timeout: 60_000,
          maxBodyLength: Infinity,
          maxContentLength: Infinity,
        });
        all = all.concat(data?.results || []);
        setProgress({ cur: Math.min(i + BATCH, files.length), total: files.length });
      }
      all.sort((a, b) => b.score - a.score);
      setResults(all);
    } catch (e) {
      Alert.alert("에러", String(e?.message || e));
    } finally {
      setLoading(false);
      setHasSearched(true);
    }
  };

  return (
    <View style={s.wrap}>
      <Text style={s.title}>🔍 CLIP 이미지 검색기</Text>

      <View style={s.row}>
        <Button title="폴더 선택(안드로이드)" onPress={pickFolderAndLoad} />
        <View style={{ width: 8 }} />
        <Button title="이미지 수동 선택" onPress={pickImages} />
        <View style={{ width: 8 }} />
        <Button title="선택 해제" onPress={clearSelection} />
      </View>

      {Platform.OS === "android" && Platform.Version >= 30 && (
        <View style={s.row}>
          <Button title="모든 파일 접근 허용(안드로이드)" onPress={openAllFilesAccessSettings} />
          <Text style={s.helper}>필요 시 설정에서 토글 후 다시 시도</Text>
        </View>
      )}

      <TextInput
        style={s.input}
        placeholder="텍스트 프롬프트 (예: cat on the sofa)"
        placeholderTextColor="#889"
        value={prompt}
        onChangeText={setPrompt}
      />

      <View style={s.row}>
        <Text style={s.label}>임계치 {threshold.toFixed(2)}</Text>
        <Slider
          style={{ flex: 1 }}
          minimumValue={0}
          maximumValue={1}
          step={0.01}
          value={threshold}
          onValueChange={setThreshold}
        />
      </View>

      <View style={s.row}>
        <Button title="검색 실행" onPress={search} disabled={loading} />
        <View style={{ width: 10 }} />
        <Text style={s.muted}>
          {files.length ? `${files.length}장 선택됨` : "선택된 이미지 없음"}
        </Text>
      </View>

      {loading && (
        <View style={s.progress}>
          <ActivityIndicator />
          <Text style={s.progressText}>
            업로드/검색 중… {progress.cur}/{progress.total}
          </Text>
        </View>
      )}

      <Text style={s.h3}>결과</Text>

      {hasSearched && !loading && results.length === 0 ? (
        <Text style={s.empty}>결과가 없습니다.</Text>
      ) : (
        <FlatList
          data={results}
          keyExtractor={(it, idx) => (it.name || "name") + "_" + idx}
          ItemSeparatorComponent={() => <View style={{ height: 8 }} />}
          renderItem={({ item }) => (
            <View style={s.card}>
              <View style={{ flex: 1 }}>
                <Text numberOfLines={1} style={s.text}>{item.name}</Text>
                <Text style={s.sub}>score: {Number(item.score).toFixed(3)}</Text>
              </View>
            </View>
          )}
        />
      )}
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { flex: 1, padding: 16, backgroundColor: "#0b0f14" },
  title: { color: "#e7eaf0", fontSize: 20, fontWeight: "700", marginBottom: 12 },
  row: { flexDirection: "row", alignItems: "center", gap: 8, marginVertical: 6 },
  input: { borderWidth: 1, borderColor: "#233", borderRadius: 10, padding: 12, color: "#e7eaf0", backgroundColor: "#10151c" },
  label: { color: "#e7eaf0", marginRight: 8 },
  h3: { color: "#e7eaf0", fontSize: 16, fontWeight: "600", marginTop: 12, marginBottom: 6 },
  card: { flexDirection: "row", gap: 12, borderWidth: 1, borderColor: "#233", backgroundColor: "#10151c", borderRadius: 12, padding: 10, alignItems: "center" },
  text: { color: "#e7eaf0" },
  sub: { color: "#9aa3ad", marginTop: 2 },
  progress: { flexDirection: "row", alignItems: "center", gap: 10, marginVertical: 8 },
  progressText: { color: "#e7eaf0" },
  muted: { color: "#9aa3ad" },
  helper: { color: "#9aa3ad", marginLeft: 8 },
  empty: { color: "#9aa3ad", paddingVertical: 12 },
});
