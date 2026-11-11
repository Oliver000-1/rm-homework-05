# 猎隼战队算法组第五次作业

固定相机下的装甲板测量和跟踪。项目用 C++17 和 CMake，实拍入口在 Ubuntu 中使用 MVS Linux SDK。沿用上次的 ONNX 模型和四关键点输出。

## 现场要记的数值

| 对象 | 需要记录 |
|---|---|
| 棋盘格 | 横向内角点数、纵向内角点数、相邻角点间距（mm）。例如横向有 10 个方格，横向内角点就有 9 个 |
| 装甲板 | 模型 1–4 号点在实物上的位置，以及这四个点对应的实际横向宽度和纵向高度（mm）。按对应点测量 |
| 相机 | 程序自动记录序列号、原图宽高和 ROI 偏移。标定与正式演示保持相同镜头、对焦和采集设置 |

## 1. 在 Ubuntu 构建

先用 MVS 客户端检查出图和曝光，关闭客户端释放相机。在 Ubuntu 本地目录构建。

```bash
sudo apt install build-essential cmake libopencv-dev usbutils
bash scripts/prepare_ubuntu.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MVS=ON
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/armor_live --list-cameras
```

MVS 默认安装路径为 `/opt/MVS`。安装在其他位置时给 CMake 传 `-DMVS_ROOT=/实际路径`。无需相机的构建可用 `-DENABLE_MVS=OFF`，视频回放仍可用。模型和 ONNX Runtime 由准备脚本检查 SHA-256。

## 2. 拍棋盘格并标定

把 `SERIAL`、`COLS`、`ROWS`、`SQUARE_MM` 替换为现场记录。建议先用 `--preview` 看程序实际取到的画面。程序打开相机时会重新设置曝光和增益，在 MVS 客户端调过的数值可通过 `--exposure-us` 和 `--gain-db` 传入。

```bash
./build/armor_live --serial SERIAL --capture-calibration --seconds 360 --output outputs/calibration
```

相机固定，移动和倾斜棋盘格。画面稳定且角点清楚时按 **S** 保存原尺寸 PNG。让棋盘覆盖中心和四周，尝试不同距离和倾角，目标是 20 张以上可用图。程序会打印本次 `calibration-frames` 目录。该目录里的 `capture.yml` 记录相机和 ROI。

```bash
./build/armor_calibrate --images outputs/calibration/会话名/calibration-frames \
  --cols COLS --rows ROWS --square-mm SQUARE_MM --output camera.yml
```

工具按文件名排序，每五张可用图留出一张验证，至少需要 12 张可用图。看 `camera.yml` 的拟合和留出重投影误差，发现某张误差明显偏大时，核对原图清晰度、角点覆盖和倾角，再用新的输出文件名重新拟合。手上只有少量照片时，可用 `--check-only --images DIR --cols COLS --rows ROWS` 验证角点检测。没有采集记录的照片生成的参数可用于离线检查。

## 3. 实时识别、测距和跟踪

先用 `./build/armor_live --serial SERIAL --seconds 60` 检查同一块装甲板正面与斜视的关键点编号。`--corner-order` 按 **左上、左下、右下、右上** 填入这四个实体角各自的模型编号。旧视频参考数据的初始顺序为 `1234`，以实物核对结果为准。

```bash
./build/armor_live --serial SERIAL --calibration camera.yml \
  --armor-width-mm WIDTH_MM --armor-height-mm HEIGHT_MM \
  --corner-order 1234 --seconds 90 --record --output outputs/tracking
```

演示时画面里放一块装甲板。其他类别入镜时，可用 `--target` 按实物类别限定，例如 `--target B3`。候选有多块时该帧按缺测处理。实时入口会核对标定的序列号、分辨率和 ROI。画面叠加四点、姿态轴、原始位置和距离、滤波位置、速度与未来 0.2 秒的位置。

- `X/Y/Z` 对应相机的右、下、前，单位为米。距离为相机光心到装甲板中心的直线距离。姿态用画面坐标轴和 CSV 中的旋转向量表示。
- `tracking.csv` 每帧记录时间、原始位置、重投影误差、滤波状态和预测。失去测量时原始列留空，短时继续预测。达到 0.5 秒后结束当前跟踪，下一次有效测量重新初始化。
- 初始测量标准差为 `20 20 80` mm。采集静止段后可将原始三轴的标准差填入 `--measurement-std-mm SX SY SZ`，并记录演示实际使用的参数。
- **S** 截图，**R** 切换录像，**Q/Esc** 退出。录像按 10 FPS 写入当前会话，单段上限约 128 MiB。CSV 保存测量的实际时间间隔。

## 4. 实拍验收和提交

1. **位置与距离**。相机固定，在近、中、远各把板静止约 10 秒，记录摆放和测距结果。对照卷尺时把装甲板放在光轴附近，以 `Z` 作为轴向深度读数。
2. **运动与滤波**。横移、停下并短暂遮挡。视频中展示原始、滤波和预测三组数据。用 `tracking.csv` 的同一个静止时间窗比较滤波前后各轴的标准差，再观察运动中的滞后。
3. **提交**。个人 Git 仓库包含源码、可重现的运行步骤、现场使用的标定和装甲板配置。另交实拍视频，展示装甲板识别、PnP 和滤波效果。

离线可用 `--video videos/armor_original.mp4` 检查识别、关键点和录像入口。CTest 覆盖已知位姿、角点与标定不匹配、可变帧间隔、遮挡和恢复。最终视频使用现场相机的连续帧。

## 2026-11-11 实验室记录

Ubuntu 22.04 上的 C++17 工程完成构建，3 项 CTest 均通过。实拍使用 MV-CA016-10UC，相机序列号 `00L28498186`，原图为 1440×1080。棋盘格为 9×6 个内角点，格距 26 mm。21 张可用照片中，17 张参与拟合，4 张留出验证。两组重投影 RMS 分别为 0.652 px 和 0.676 px。标定结果见 [camera-00L28498186.yml](camera-00L28498186.yml)。

实拍的装甲板类别为 B1，四点建模宽 233 mm、高 56 mm，点序为 `1234`。同一相机和成像设置下，运行下面的命令，按 **R** 录像。

```bash
./build/armor_live --serial 00L28498186 \
  --calibration camera-00L28498186.yml \
  --armor-width-mm 233 --armor-height-mm 56 \
  --corner-order 1234 --target B1 \
  --exposure-us 20000 --gain-db 6 \
  --seconds 120 --output outputs/tracking
```

演示选取两段测距和一段遮挡恢复。近、远片段的 `Z` 读数约为 1.51 m 和 2.07 m。遮挡时，画面显示 `prediction_only`，原始测量留空，短时间内继续预测。装甲板重新进入视野后恢复测量。

在测距轮次的 CSV 时间 42–47 s 内，取 46 个有测量修正的同帧样本比较。原始与滤波后的 `Z` 标准差分别为 8.10 mm 和 7.39 mm，`X` 和 `Y` 在该段基本持平。测距轮次共处理 814 帧，有效位姿 185 帧。遮挡轮次共处理 559 帧，有效位姿 248 帧。画面出现出框或位姿校验失败时，有效测量会中断。

- [测距跟踪数据](evidence/distance/tracking.csv)与[运行报告](evidence/distance/report.txt)
- [遮挡跟踪数据](evidence/occlusion/tracking.csv)与[运行报告](evidence/occlusion/report.txt)
