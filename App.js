import React, { useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Button,
  FlatList,
  Image,
  StyleSheet,
  Text,
  TextInput,
  View,
  PermissionsAndroid,
  Platform,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import * as MediaLibrary from "expo-media-library";
import * as FileSystem from "expo-file-system";
import Slider from "@react-native-community/slider";
import axios from "axios";

// ⚠️ 에뮬레이터: 10.0.2.2 / 실기기: PC 로컬 IP
// 예) const BASE_URL = "http://192.168.0.23:8000";
const BASE_URL = "http://10.0.2.2:8000";

export default function App() {
  // 파일 목록: { uri, name, mimeType, size? }
  const [files, setFiles] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [threshold, setThreshold] = useState(0.3);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState({ cur: 0, total: 1 });
  const [results, setResults] = useState([]); // [{ name, score }]
  const [permission, requestPermission] = MediaLibrary.usePermissions();

  const preview = useMemo(() => files.slice(0, 50), [files]);
  const BATCH = 8; // 너무 크면 서버 500 발생 → 8~16 권장

  // content:// → file:// 캐시 복사 (서버가 읽을 수 있게)
  const toUploadable = async (f) => {
    if (f?.uri?.startsWith("content://")) {
      const safe = (f.name || "image.jpg").replace(/[^a-zA-Z0-9._-]/g, "_");
      const dest = `${FileSystem.cacheDirectory}up-${Date.now()}-${safe}`;
      await FileSystem.copyAsync({ from: f.uri, to: dest });
      return { ...f, uri: dest };
    }
    return f;
  };

  /** 모든 사진 불러오기 (MediaStore) */
  const loadAllPhotos = async () => {
    try {
      // 권한 확인/요청
      if (!permission || permission.status !== "granted") {
        const res = await (permission?.canAskAgain
          ? requestPermission()
          : Promise.resolve(permission));
        if (!res || res.status !== "granted") {
          Alert.alert("권한 필요", "사진 접근 권한을 허용해주세요.");
          return;
        }
      }
      if (Platform.OS === "android" && Platform.Version >= 29) {
        await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_MEDIA_LOCATION
        ).catch(() => {});
      }

      const pageSize = 1000;
      let endCursor = null,
        hasNext = true;
      const assets = [];

      while (hasNext && assets.length < 5000) {
        const res = await MediaLibrary.getAssetsAsync({
          mediaType: "photo",
          first: pageSize,
          after: endCursor,
          sortBy: [["creationTime", false]], // 최신순
        });
        assets.push(...res.assets);
        endCursor = res.endCursor;
        hasNext = res.hasNextPage;
      }

      const detailed = await Promise.all(
        assets.map(async (a) => {
          const info = await MediaLibrary.getAssetInfoAsync(a);
          return {
            uri: info.localUri || a.uri,
            name: info.filename || `${a.id}.jpg`,
            mimeType: "image/jpeg",
          };
        })
      );

      setFiles(detailed);
      setResults([]);
      Alert.alert("완료", `사진 ${detailed.length}장을 불러왔습니다.`);
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
      setFiles(
        (res.assets || []).map((a) => ({
          uri: a.uri,
          name: a.name || "image.jpg",
          mimeType: a.mimeType || "image/jpeg",
          size: a.size,
        }))
      );
      setResults([]);
    } catch (e) {
      Alert.alert("선택 오류", String(e?.message || e));
    }
  };

  const clearSelection = () => {
    setFiles([]);
    setResults([]);
  };

  /** 서버 검색 (배치 업로드) */
  const search = async () => {
    if (!files.length) {
      Alert.alert("안내", "사진을 불러오거나 선택하세요.");
      return;
    }
    if (!prompt.trim()) {
      Alert.alert("안내", "텍스트 프롬프트를 입력하세요.");
      return;
    }

    setLoading(true);
    setResults([]);
    setProgress({ cur: 0, total: files.length });

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
            uri: f.uri,
            name: f.name || "image.jpg",
            type: f.mimeType || "image/jpeg",
          });
        }

        const { data } = await axios.post(`${BASE_URL}/search`, form, {
          // ⚠️ Content-Type 헤더를 수동으로 지정하지 마세요 (boundary 깨짐)
          timeout: 60_000,
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
        });

        all = all.concat(data?.results || []);
        setProgress({
          cur: Math.min(i + BATCH, files.length),
          total: files.length,
        });
      }
      all.sort((a, b) => b.score - a.score);
      setResults(all);
    } catch (e) {
      const status = e?.response?.status;
      const body = e?.response?.data;
      const msg =
        body && typeof body !== "string" ? JSON.stringify(body) : body || e.message || String(e);
      Alert.alert(`에러${status ? " " + status : ""}`, msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={s.wrap}>
      <Text style={s.title}>🔍 CLIP 이미지 검색기</Text>

      <View style={s.row}>
        <Button title="모든 사진 불러오기" onPress={loadAllPhotos} />
        <View style={{ width: 8 }} />
        <Button title="이미지 수동 선택" onPress={pickImages} />
        <View style={{ width: 8 }} />
        <Button title="선택 해제" onPress={clearSelection} />
      </View>

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
      </View>

      {loading && (
        <View style={s.progress}>
          <ActivityIndicator />
          <Text style={s.progressText}>
            업로드/검색 중… {progress.cur}/{progress.total}
          </Text>
        </View>
      )}

      <Text style={s.h3}>선택된/불러온 사진 ({files.length}장)</Text>
      <FlatList
        data={preview}
        keyExtractor={(it, idx) => it.uri + "_" + idx}
        ItemSeparatorComponent={() => <View style={{ height: 8 }} />}
        renderItem={({ item }) => (
          <View style={s.card}>
            <Image source={{ uri: item.uri }} style={s.thumb} />
            <View style={{ flex: 1 }}>
              <Text numberOfLines={1} style={s.text}>
                {item.name || item.uri}
              </Text>
              {!!item.size && (
                <Text style={s.sub}>{(item.size / 1024 / 1024).toFixed(2)} MB</Text>
              )}
            </View>
          </View>
        )}
      />

      <Text style={s.h3}>결과 ({results.length}장)</Text>
      <FlatList
        data={results}
        keyExtractor={(it, idx) => (it.name || "name") + "_" + idx}
        ItemSeparatorComponent={() => <View style={{ height: 8 }} />}
        renderItem={({ item }) => {
          const f = files.find((x) => (x.name || "") === item.name);
          return (
            <View style={s.card}>
              {!!f && <Image source={{ uri: f.uri }} style={s.thumb} />}
              <View style={{ flex: 1 }}>
                <Text numberOfLines={1} style={s.text}>
                  {item.name}
                </Text>
                <Text style={s.sub}>score: {item.score.toFixed(3)}</Text>
              </View>
            </View>
          );
        }}
      />
    </View>
  );
}

const s = StyleSheet.create({
  wrap: { flex: 1, padding: 16, backgroundColor: "#0b0f14" },
  title: { color: "#e7eaf0", fontSize: 20, fontWeight: "700", marginBottom: 12 },
  row: { flexDirection: "row", alignItems: "center", gap: 8, marginVertical: 6 },
  input: {
    borderWidth: 1,
    borderColor: "#233",
    borderRadius: 10,
    padding: 12,
    color: "#e7eaf0",
    backgroundColor: "#10151c",
  },
  label: { color: "#e7eaf0", marginRight: 8 },
  h3: { color: "#e7eaf0", fontSize: 16, fontWeight: "600", marginTop: 12, marginBottom: 6 },
  card: {
    flexDirection: "row",
    gap: 12,
    borderWidth: 1,
    borderColor: "#233",
    backgroundColor: "#10151c",
    borderRadius: 12,
    padding: 10,
    alignItems: "center",
  },
  thumb: { width: 64, height: 64, borderRadius: 8, borderWidth: 1, borderColor: "#233" },
  text: { color: "#e7eaf0" },
  sub: { color: "#9aa3ad", marginTop: 2 },
  progress: { flexDirection: "row", alignItems: "center", gap: 10, marginVertical: 8 },
  progressText: { color: "#e7eaf0" },
});
