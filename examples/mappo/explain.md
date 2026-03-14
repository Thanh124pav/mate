Dưới đây là **giải thích chi tiết từng dòng** của đoạn code bạn cung cấp. Đây là một **mô hình mạng nơ-ron tùy chỉnh** cho thuật toán **MAPPO** (Multi-Agent Proximal Policy Optimization) trong **Ray RLlib**, sử dụng **PyTorch** và **RNN (LSTM)** để xử lý quan sát cục bộ và trạng thái toàn cục.

---

### **Import các thư viện cần thiết**

```python
from collections import OrderedDict
```
> Dùng để lưu thứ tự các phần tử trong dict (giữ nguyên thứ tự khi thêm key).

```python
import numpy as np
from gym import spaces
```
> - `np`: xử lý mảng số.
> - `spaces`: định nghĩa không gian quan sát/hành động trong Gym (Box, Discrete, Dict...).

```python
from ray.rllib.models import ModelCatalog
```
> `ModelCatalog`: đăng ký mô hình tùy chỉnh để RLlib có thể sử dụng.

```python
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
```
> Lớp cơ sở cho mô hình **RNN** dùng PyTorch trong RLlib.

```python
from ray.rllib.utils.framework import try_import_torch
```
> Hàm an toàn để import `torch` và `nn` (neural network module), tránh lỗi nếu không có.

```python
from examples.utils import SimpleRNN, get_space_flat_size, orthogonal_initializer
```
> Các hàm tiện ích:
> - `SimpleRNN`: lớp RNN đơn giản (MLP + LSTM).
> - `get_space_flat_size`: tính kích thước phẳng của một `gym.space`.
> - `orthogonal_initializer`: khởi tạo trọng số theo phân phối orthogonal.

```python
torch, nn = try_import_torch()
```
> Import `torch` và `torch.nn` một cách an toàn.

---

## **Định nghĩa lớp mô hình: `MAPPOModel`**

```python
class MAPPOModel(TorchRNN, nn.Module):
```
> Kế thừa từ:
> - `TorchRNN`: hỗ trợ RNN trong RLlib.
> - `nn.Module`: lớp cơ sở của PyTorch.

---

### **Hàm `__init__` – Khởi tạo mô hình**

```python
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        # Extra MAPPOModel arguments
        actor_hiddens=None,
        actor_hidden_activation='tanh',
        critic_hiddens=None,
        critic_hidden_activation='tanh',
        lstm_cell_size=256,
        **kwargs,
    ):
```
> Các tham số:
> - `obs_space`, `action_space`: không gian quan sát và hành động.
> - `num_outputs`: số lượng hành động (discrete).
> - `model_config`, `name`: cấu hình và tên mô hình.
> - Các tham số tùy chỉnh cho MAPPO.

```python
        if actor_hiddens is None:
            actor_hiddens = [256, 256]
```
> Mặc định: 2 tầng ẩn cho **actor**, mỗi tầng 256 đơn vị.

```python
        if critic_hiddens is None:
            critic_hiddens = [256, 256]
```
> Tương tự cho **critic**.

```python
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
```
> Gọi constructor của `nn.Module` và `TorchRNN`.

---

### **Xử lý không gian quan sát dạng Dict**

```python
        assert hasattr(obs_space, 'original_space') and isinstance(
            obs_space.original_space, spaces.Dict
        )
```
> Kiểm tra: `obs_space` phải có `original_space` là `Dict` (do RLlib bọc lại).

```python
        original_space = obs_space.original_space
        self.local_obs_space = original_space['obs']
        self.global_state_space = original_space['state']
```
> Trích xuất:
> - `'obs'`: quan sát **cục bộ** của agent.
> - `'state'`: trạng thái **toàn cục** (dùng cho critic trong CTDE).

```python
        if 'action_mask' in original_space.spaces:
            self.action_mask_space = original_space['action_mask']
            self.has_action_mask = True
        else:
            self.action_mask_space = None
            self.has_action_mask = False
```
> Kiểm tra có **action masking** không (ví dụ: trong môi trường chỉ cho phép một số hành động).

---

### **Tính toán kích thước phẳng của các không gian**

```python
        self.flat_obs_dim = get_space_flat_size(self.obs_space)
```
> Tổng chiều của đầu vào đã được làm phẳng (do RLlib tự động flatten).

```python
        self.space_dims = OrderedDict(
            [(key, get_space_flat_size(subspace)) for key, subspace in original_space.items()]
        )
```
> Tạo dict: `{'obs': d1, 'state': d2, 'action_mask': d3}` → lưu kích thước từng phần.

```python
        indices = np.cumsum([0, *self.space_dims.values()])
        self.flat_obs_slices = OrderedDict(
            [
                (key, slice(indices[i], indices[i + 1]))
                for i, key in enumerate(self.space_dims.keys())
            ]
        )
```
> Tạo **slice** để trích xuất từng phần từ vector phẳng:
> - Ví dụ: `obs` từ vị trí 0 đến d1, `state` từ d1 đến d1+d2,...

```python
        self.local_obs_dim = self.space_dims['obs']
        self.local_obs_slice = self.flat_obs_slices['obs']
        self.global_state_dim = self.space_dims['state']
        self.global_state_slice = self.flat_obs_slices['state']
```
> Lưu riêng kích thước và slice cho `obs` và `state`.

```python
        self.action_dim = get_space_flat_size(self.action_space)
```
> Kích thước hành động (thường là số hành động discrete).

```python
        if self.has_action_mask:
            self.action_mask_slice = self.flat_obs_slices['action_mask']
            assert self.space_dims['action_mask'] == num_outputs
        else:
            self.action_mask_slice = None
```
> Nếu có action mask → kiểm tra kích thước phải bằng số hành động.

---

### **Lưu cấu hình mạng**

```python
        self.actor_hiddens = actor_hiddens or []
        self.critic_hiddens = critic_hiddens or list(self.actor_hiddens)
        self.actor_hidden_activation = actor_hidden_activation
        self.critic_hidden_activation = critic_hidden_activation
        self.lstm_cell_size = lstm_cell_size
```
> Lưu các siêu tham số cho actor/critic.

---

### **Tạo mạng Actor và Critic (dùng SimpleRNN)**

```python
        self.actor = SimpleRNN(
            name='actor',
            input_dim=self.local_obs_dim,
            hidden_dims=self.actor_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=num_outputs,
            activation=self.actor_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=0.01),
        )
```
> **Actor**:
> - Input: quan sát cục bộ.
> - MLP → LSTM → output logits cho hành động.
> - Khởi tạo orthogonal, output scale nhỏ (0.01) → ổn định học.

```python
        self.critic = SimpleRNN(
            name='critic',
            input_dim=self.global_state_dim,
            hidden_dims=self.critic_hiddens,
            cell_size=self.lstm_cell_size,
            output_dim=1,
            activation=self.critic_hidden_activation,
            output_activation=None,
            hidden_weight_initializer=orthogonal_initializer(scale=1.0),
            output_weight_initializer=orthogonal_initializer(scale=1.0),
        )
```
> **Critic**:
> - Input: trạng thái toàn cục.
> - Output: giá trị state-value (1 chiều).
> - Dùng cùng LSTM để chia sẻ bộ nhớ thời gian.

---

### **Trả về trạng thái ban đầu của RNN**

```python
    def get_initial_state(self):
        return [*self.actor.get_initial_state(), *self.critic.get_initial_state()]
```
> Gộp trạng thái ẩn ban đầu của **actor LSTM** và **critic LSTM**.

---

### **Forward pass cho RNN (xử lý chuỗi)**

```python
    def forward_rnn(self, inputs, state, seq_lens):
```
> - `inputs`: [B, T, flat_dim] – batch x time x features.
> - `state`: danh sách trạng thái ẩn (h, c cho actor + critic).
> - `seq_lens`: độ dài chuỗi thực tế (dùng trong padding).

```python
        assert inputs.size(-1) == self.flat_obs_dim
```
> Kiểm tra kích thước đầu vào.

```python
        local_obs = inputs[..., self.local_obs_slice]
```
> Trích xuất quan sát cục bộ từ vector phẳng.

```python
        actor_state_in = state[:2]
        action_out, actor_state_out = self.actor(local_obs, actor_state_in)
```
> Chạy actor RNN → ra **logits hành động** và **trạng thái mới**.

```python
        if self.has_action_mask:
            action_mask = inputs[..., self.action_mask_slice].clamp(min=0.0, max=1.0)
            inf_mask = torch.log(action_mask).clamp_min(min=torch.finfo(action_out.dtype).min)
            action_out = action_out + inf_mask
```
> **Action masking**:
> - `action_mask`: [0,1] → 1 nếu hành động hợp lệ.
> - Lấy log → 0 nếu hợp lệ, -inf nếu không.
> - Cộng vào logits → hành động không hợp lệ có xác suất ~0.

```python
        global_state = inputs[..., self.global_state_slice]
        critic_state_in = state[2:]
        _, critic_state_out = self.critic(global_state, critic_state_in, features_only=True)
```
> Chạy critic RNN, nhưng **chỉ lấy features** (không cần output ở đây).
> - Lưu `last_features` để dùng trong `value_function()`.

```python
        return action_out, [*actor_state_out, *critic_state_out]
```
> Trả về:
> - `action_out`: logits hành động.
> - Trạng thái mới của cả actor và critic.

---

### **Hàm giá trị (Value Function)**

```python
    def value_function(self):
        assert self.critic.last_features is not None, 'must call forward() first'
```
> Phải gọi `forward_rnn` trước.

```python
        return self.critic.output(self.critic.last_features).reshape(-1)
```
> Dùng layer output cuối của critic để tính **state value**, làm phẳng thành vector [B*T].

---

### **Đăng ký mô hình tùy chỉnh**

```python
ModelCatalog.register_custom_model('MAPPOModel', MAPPOModel)
```
> Đăng ký để RLlib có thể dùng:
> ```python:disable-run
> "model": {"custom_model": "MAPPOModel"}
> ```

---

## **Tóm tắt chức năng tổng thể**

| Thành phần | Chức năng |
|-----------|----------|
| **Actor** | Dùng **quan sát cục bộ** + LSTM → dự đoán hành động |
| **Critic** | Dùng **trạng thái toàn cục** + LSTM → ước lượng giá trị |
| **CTDE** | Centralized Training (critic thấy toàn cục), Decentralized Execution (actor chỉ thấy cục bộ) |
| **Action Masking** | Hỗ trợ môi trường có hành động không hợp lệ |
| **RNN** | Xử lý dữ liệu tuần tự (PPO với frame stacking hoặc môi trường có thời gian) |

---

## **Ứng dụng thực tế**

Mô hình này thường dùng trong:
- **StarCraft II (SMAC)**
- **MPE (Multi-Agent Particle Env)**
- **Các game RTS, MOBA** có trạng thái toàn cục và quan sát cục bộ.

---

**Kết luận**: Đây là một **mô hình MAPPO chuẩn với RNN**, hỗ trợ **CTDE**, **action masking**, và **tích hợp mượt mà với Ray RLlib**.
```

