# readme

对于一些边端设备，需要进一步将onnx模型转成对应芯片适配的模型，anchor需要手动输入的情况



编译

```shell
mkdir build && cd build
cmake .. && make
```

运行

```shell
./demo --model_path --image_path
```