import torch
import torch.nn.functional as F

import torch.utils
from torch.utils.tensorboard import SummaryWriter
import torch.utils.tensorboard
import matplotlib.pyplot as plt

import math


@torch.no_grad()
def log_condition(
    step: int, 
    model: torch.nn.Module, 
    writer: SummaryWriter, 
    loglist: set,
    loghist: bool=False
):
    for name, p in model.named_parameters():
        if "weight" in name and (not "LayerNorm" in name) and (not "embeddings" in name) and (not "prediction" in name):
            t = p.data
            
            if "heads" in name:
                
                for i in range(3):
                    tt = t.view(3, -1, t.size()[1])[i]
                    s = torch.linalg.svdvals(tt)
                
                    sn = s * s / (s.norm() ** 2) 
                    # erank = (- sn * sn.log()).sum().exp()
                    
                    # Calculate effective rank using the new method
                    effective_rank = (s ** 2).sum() / (s[0] ** 2) if s[0] != 0 else 0
                    
                    writer.add_scalar(f"cond1/{name}/{i}", s[0] / s[-1], step)
                    writer.add_scalar(f"cond2/{name}/{i}", s[0] / s[1], step)
                    
                    writer.add_scalar(f"erank/{name}/{i}", effective_rank, step)
                    
                    writer.add_scalar(f"eigmax/{name}{i}", s[0], step)
            else:
                s = torch.linalg.svdvals(t)
                
                sn = s * s / (s.norm() ** 2) 
                erank = (- sn * sn.log()).sum().exp()
                
                # Calculate effective rank using the new method
                effective_rank = (s ** 2).sum() / (s[0] ** 2) if s[0] != 0 else 0
                
                writer.add_scalar(f"cond1/{name}", s[0] / s[-1], step)
                writer.add_scalar(f"cond2/{name}", s[0] / s[1], step)
                
                writer.add_scalar(f"erank/{name}", effective_rank, step)
                
                writer.add_scalar(f"eigmax/{name}", s[0], step)
            
            if loghist:
                writer.add_histogram(name, s, step, bins=100)
                
        
        pass

@torch.no_grad()
def log_model_grad_norm(
    step: int,
    model: torch.nn.Module,
    writer: SummaryWriter
):
    grad = 0
    for p in model.parameters():
        if not p.grad is None:
            grad += p.grad.norm().item() ** 2
    
    writer.add_scalar("model_grad_norm", grad ** 0.5, step)
    return grad ** 0.5

@torch.no_grad()
def log_module_grad_norm(step, model, writer, loglist):
    for name, p in model.named_parameters():
        if name in loglist and p.grad is not None:
            g = p.grad
            
            if "heads" in name:
                g = g.view(3, -1, 768)[0]            
            
            writer.add_scalar(f"grad norm/{name}", torch.norm(g), step) 

@torch.no_grad()
def log_params(
    step: int, 
    model: torch.nn.Module, 
    logdir: str, 
    loglist: set
):
    d = {}
    for name, p in model.named_parameters():
        if name in loglist:
            d[name] = p.data.detach()
            
    torch.save(d, f"{logdir}/params_step_{step}")
    
    pass

@torch.no_grad()
def log_grads(
    step: int, 
    model: torch.nn.Module, 
    logdir: str, 
    loglist: set
): 
    d = {}
    for name, p in model.named_parameters():
        if name in loglist:
            d[name] = p.grad.detach()
            
    torch.save(d, f"{logdir}/grads_step_{step}")
    
    pass

@torch.no_grad()
def log_test_loss(
    step: int,
    model: torch.nn.Module,
    writer: SummaryWriter, 
    test_loader
):
    t_loss = 0
    for i, batch in enumerate(test_loader):
        loss, _ = model(**batch)
        t_loss += loss.item()
        
    writer.add_scalar("test_loss", t_loss / len(test_loader), step)
    

def log_eigenvalue(step, model, writer, loglist):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in loglist:
                t = p.data
                
                if "heads" in name:
                    t = t.view(3, -1, t.size()[1])[0]
                
                s = torch.linalg.svdvals(t)                
                
                writer.add_scalar(f"EigenValue Max/{name}", s[0], step)
                writer.add_scalar(f"EigenValue Second/{name}", s[1], step)           
                #writer.add_scalar(f"EigenValue Min/{name}", s[-1], step)

def log_hessian(model, batch, optimizer, log_list, writer, step, iter_num=100):
    params = []
    param_names = []  # To store the names of the parameters
    for name, p in model.named_parameters():
        if name in log_list:
            params.append(p)
            param_names.append(name)  # Save the parameter name
    
    model.eval()
    optimizer.zero_grad()
    batch = {k: v[0:16] for k, v in batch.items()}  # Reduce batch size for Hessian computation
    loss, _ = model(**batch)

    V = [torch.ones_like(p, device=p.device) for p in params]
    V = [v / torch.norm(v) for v in V]

    for _ in range(iter_num):
        model.zero_grad()
        grad = torch.autograd.grad(loss, params, create_graph=True)
        grad = [g / torch.norm(g) for g in grad]
        Hv = torch.autograd.grad(grad, params, V, only_inputs=True, retain_graph=True)
        max_eigs = torch.tensor([torch.sum(h * v) for h, v in zip(Hv, V)], device=params[0].device)
        V = [v / torch.norm(v) for v in Hv]

    # Create a dictionary to store max eigenvalues with parameter names
    for name, eig, vec in zip(param_names, max_eigs, V):
        writer.add_scalar(f"Hessian Max/{name}", abs(eig.item()), step)
        
    model.train() 
    # for name, eig, vec in zip(param_names, max_eigs, V):
    #     if name == "bert.encoder.layers.0.attention.self.heads.0.weight":
    #         return loss.item(), abs(eig.item())


def log_ntk(model, batch, optimizer, ntk_set: set, step, writer):
    gdict = {}
    ndict = {}
    losses = []
    
    # ========================= ntk =========================
    
    model.eval()
    for i in range(64):
        b = {k: v[i: i + 1] for k, v in batch.items()}
            
        optimizer.zero_grad()

        loss, logit = model(**b)

        losses.append(loss.item())
        label_list = []
        for i, l in enumerate(b["labels"][0]):
            if l > 0 :
                label_list.append((i, l.item()))

        x = torch.nn.functional.softmax(logit[0])

        t = torch.tensor(0, dtype=x.dtype, device=x.device)
        for l in label_list:
            t += x[l[0]][l[1]]
        t.backward()
            
        for name, p in model.named_parameters():
            if (name in ntk_set) and (not p.grad is None) and p.requires_grad:
                glist = gdict.get(name)
                if glist is None:                    
                    gdict[name] = [p.grad.detach().clone().flatten()]
                else:                    
                    glist.append(p.grad.detach().clone().flatten())
        
    for name, glist in gdict.items():
        glist = torch.stack(glist)
        # Normalize each gradient vector to have a norm of 1
        glist = torch.nn.functional.normalize(glist, p=2, dim=1)
        ntk_matrix = glist @ glist.T
        ndict[name] = ntk_matrix
        
        # Calculate the eigenvalues and eigenvectors of the NTK matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(ntk_matrix)

        # Sort eigenvalues in ascending order
        eigenvalues = eigenvalues.sort().values

        # Get the maximum eigenvalue
        max_eigenvalue = eigenvalues[-1].item()  # Largest eigenvalue
        second_max_eigenvalue = eigenvalues[-2].item()  # Second-largest eigenvalue
       
        # Log the maximum eigenvalue to TensorBoard
        writer.add_scalar(f'NTK_Max_Eigenvalue/{name}', max_eigenvalue, step)
        writer.add_scalar(f'NTK_Second_Max_Eigenvalue/{name}', second_max_eigenvalue, step)
    
    model.train()


def log_effective_rank(step, model, writer, loglist):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in loglist:
                t = p.data
                
                if "heads" in name:
                    t = t.view(3, -1, t.size()[1])[0]
                
                s = torch.linalg.svdvals(t)       
                
                # Normalize the singular values to get a probability distribution
                normalized_singular_values = s / torch.sum(s)
                
                # Calculate the entropy
                epsilon=1e-10
                entropy = -torch.sum(normalized_singular_values * torch.log(normalized_singular_values + epsilon))
                
                # The effective rank is defined as the exponent of the entropy
                eff_rank = torch.exp(entropy)     
                
                writer.add_scalar(f'effective_rank/{name}', eff_rank, step)
                
class attention_after_qk_hook:
    def __init__(
        self, 
        writer: torch.utils.tensorboard.SummaryWriter, 
        name: str,
        max_logsteps: int=5000,
        min_logsteps: int=0
    ) -> None:
        self.name = name
        self.writer = writer
        self.max_logsteps = max_logsteps
        self.min_logsteps = min_logsteps
        self.steps = 0
        
    @torch.no_grad()
    def __call__(self, m: torch.nn.Module, input, output, loghist=True) -> torch.Any:
        if self.min_logsteps <= self.steps < self.max_logsteps:
            t = output[0][0].abs()
            t1 = (t.max() / t.mean()).item()
            t2 = t.max().item()
            
            if self.steps % 25 == 0 and loghist:
                self.writer.add_histogram(f"atten_logit_dis/{self.name}", t.flatten(), self.steps)
            self.writer.add_scalar(f"atten_logit/{self.name}", t1, self.steps)
            self.writer.add_scalar(f"atten_logit_abs/{self.name}", t2, self.steps)
        self.steps += 1
    

def log_average_token_similarity(model, batch, writer, step):
    with torch.no_grad():
        # Forward pass to get the outputs
        outputs = []
        h_ = model.bert.embeddings(batch['input_ids'])
        outputs.append(h_)
        for i in range(12):
            h_ = model.bert.encoder.layers[i](h_, batch['attention_mask'])
            outputs.append(h_)

        # Calculate and log average pairwise cosine similarity for each layer
        for layer_idx, layer_output in enumerate(outputs):
            # Get the output for the first sequence in the batch
            first_seq_output = layer_output[0]  # Shape: [128, 768]

            # Compute cosine similarity matrix
            cosine_similarity = F.cosine_similarity(first_seq_output.unsqueeze(0), first_seq_output.unsqueeze(1), dim=-1)  # Shape: [128, 128]

            # Average pairwise cosine similarity
            avg_cosine_similarity = cosine_similarity.mean().item()

            # Log to TensorBoard using distinct tags for each layer
            writer.add_scalar(f'Average_Cosine_Similarity/Layer_{layer_idx}', avg_cosine_similarity, step)

def log_average_first_token_similarity(model, batch, writer, step):
    with torch.no_grad():
        # Forward pass to get the outputs
        outputs = []
        h_ = model.bert.embeddings(batch['input_ids'])
        outputs.append(h_)
        for i in range(12):
            h_ = model.bert.encoder.layers[i](h_, batch['attention_mask'])
            outputs.append(h_)

        # Calculate and log average pairwise similarity for the first tokens in each sequence
        for layer_idx, layer_output in enumerate(outputs):
            # Get the output for the first tokens in the batch
            first_tokens_output = layer_output[:, 0]  # Shape: [64 (batch size), 768 (token embedding)]

            # Compute cosine similarity matrix for the first tokens
            cosine_similarity = F.cosine_similarity(first_tokens_output.unsqueeze(1), first_tokens_output.unsqueeze(0), dim=-1)  # Shape: [64, 64]

            # Average pairwise similarity
            avg_similarity = cosine_similarity.mean().item()

            # Log to TensorBoard using distinct tags for each layer
            writer.add_scalar(f'Average_First_Token_Similarity/Layer_{layer_idx}', avg_similarity, step)

def log_average_token_norm(model, batch, writer, step):
    with torch.no_grad():
        # Forward pass to get the outputs
        outputs = []
        h_ = model.bert.embeddings(batch['input_ids'])
        outputs.append(h_)
        for i in range(12):
            h_ = model.bert.encoder.layers[i](h_, batch['attention_mask'])
            outputs.append(h_)

        # Calculate and log average norm for the first sequence across all layers
        for layer_idx, layer_output in enumerate(outputs):
            # Get the output for the first sequence in the batch
            first_seq_output = layer_output[0]  # Shape: [128, 768]

            # Calculate the norm for all tokens in the first sequence (L2 norm)
            all_token_norms = first_seq_output.norm(dim=-1)  # Shape: [128]

            # Average norm across all tokens in the first sequence
            avg_norm = all_token_norms.mean().item()

            # Log to TensorBoard using distinct tags for each layer
            writer.add_scalar(f'Average_Token_Norm/Layer_{layer_idx}', avg_norm, step)


@torch.no_grad()
def log_param_sim(model, step, writer):
    layer_list = [f'layers.{i}' for i in range(12)]
    # layer_list = ['decoders.0', 'decoders.3', 'decoders.7', 'decoders.11']
    module_list = ['attn.key.weight', 'feed_fwd.0.weight', 'attn.proj.weight']
    param_eigs = [{} for i in range(12)]
    for name, p in model.named_parameters():
        for i in range(12):
            if layer_list[i] in name and 'heads.0.weight' in name:                 
                u, _, v = torch.linalg.svd(p.data, full_matrices=False)
                max_eigenvec_param = u[:, 0] 
                param_eigs[i]['head'] = max_eigenvec_param
            if layer_list[i] in name and 'attention.dense.weight' in name:
                u, _, v = torch.linalg.svd(p.data, full_matrices=False)
                max_eigenvec_param = u[:, 0] 
                param_eigs[i]['attn_dense'] = max_eigenvec_param        
            if layer_list[i] in name and 'ffn.dense_1.weight' in name:
                u, _, v = torch.linalg.svd(p.data, full_matrices=False)
                max_eigenvec_param = u[:, 0] 
                param_eigs[i]['ffn'] = max_eigenvec_param        
                
    for i in range(11):
        eig1, eig2 = param_eigs[i]['head'], param_eigs[i+1]['head']
        sim = (eig1 @ eig2.T) / eig1.norm() / eig2.norm()     
        writer.add_scalar(f'Param Sim head/layer {i}', abs(sim), step)
        
        eig1, eig2 = param_eigs[i]['attn_dense'], param_eigs[i+1]['attn_dense']
        sim = (eig1 @ eig2.T) / eig1.norm() / eig2.norm()     
        writer.add_scalar(f'Param Sim attn_dense/layer {i}', abs(sim), step)
        
        eig1, eig2 = param_eigs[i]['ffn'], param_eigs[i+1]['ffn']
        sim = (eig1 @ eig2.T) / eig1.norm() / eig2.norm()     
        writer.add_scalar(f'Param Sim ffn/layer {i}', abs(sim), step)
    

def log_projection_alignment(step, model, writer, loglist):
    for name, param in model.named_parameters():
            if name in loglist:
                W = param.detach().cpu()
                delta_W = param.grad.detach().cpu()

                # Perform SVD on the parameter matrix
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                U = U[:, :len(S)]  # 左奇异向量, 维度 (192, k)
                Vt = Vt[:len(S), :]  # 右奇异向量的转置, 维度 (k, 768)

                # 2. 计算梯度矩阵 G 在每个奇异向量方向上的投影
                projections = []
                projection_energies = []
                for i in range(len(S)):
                    u_i = U[:, i]  # 左奇异向量 u_i，shape: (192,)
                    v_i = Vt[i, :]  # 右奇异向量 v_i，shape: (768,)
                    projection = torch.dot(u_i, delta_W @ v_i.T)  # 修正的计算公式 u_i^T G v_i
                    projections.append(abs(projection))
                    projection_energies.append((projection ** 2).item())  # 保存每个投影的平方 (能量)

                # 3. 计算投影能量总和
                total_energy = sum(projection_energies)

                # 4. 计算投影能量集中在前 k 个奇异向量上的比例 (比如 k=10)
                # Calculate effective rank using the new method
                effective_rank = (S ** 2).sum() / (S[0] ** 2) if S[0] != 0 else 0
                effective_rank = math.ceil(effective_rank)
                top_k_energy = sum(projection_energies[:effective_rank])
                concentration_ratio = top_k_energy / total_energy

                # 记录 concentration_ratio 到 TensorBoard
                writer.add_scalar(f"concentration_ratio/{name}", concentration_ratio, step)

def log_grad_concentration_ratio(step, model, writer, loglist):
    for name, param in model.named_parameters():
            if name in loglist:
                W = param.detach().cpu()
                delta_W = param.grad.detach().cpu()

                # Perform SVD on the parameter matrix
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                U = U[:, :len(S)]  # 左奇异向量, 维度 (192, k)
                Vt = Vt[:len(S), :]  # 右奇异向量的转置, 维度 (k, 768)

                # 2. 计算梯度矩阵 G 在每个奇异向量方向上的投影
                projections = []
                projection_energies = []
                for i in range(len(S)):
                    u_i = U[:, i]  # 左奇异向量 u_i，shape: (192,)
                    v_i = Vt[i, :]  # 右奇异向量 v_i，shape: (768,)
                    projection = torch.dot(u_i, delta_W @ v_i.T)  # 修正的计算公式 u_i^T G v_i
                    projections.append(abs(projection))
                    projection_energies.append((projection ** 2).item())  # 保存每个投影的平方 (能量)

                # 3. 计算投影能量总和
                total_energy = sum(projection_energies)

                # 4. 计算投影能量集中在前 k 个奇异向量上的比例 (比如 k=10)
                # Calculate effective rank using the new method
                effective_rank = (S ** 2).sum() / (S[0] ** 2) if S[0] != 0 else 0
                effective_rank = math.ceil(effective_rank)
                top_k_energy = sum(projection_energies[:effective_rank])
                concentration_ratio = top_k_energy / total_energy

                # 记录 concentration_ratio 到 TensorBoard
                writer.add_scalar(f"concentration_ratio/{name}", concentration_ratio, step)


def log_grad_projection_direction(step, model, writer, loglist):
    for name, param in model.named_parameters():
            if name in loglist:
                
                    
                W = param.detach().cpu()
                delta_W = -(param.grad.detach().cpu())
                    

                # Perform SVD on the parameter matrix
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                U = U[:, :len(S)]  # 左奇异向量, 维度 (192, k)
                Vt = Vt[:len(S), :]  # 右奇异向量的转置, 维度 (k, 768)

                # 2. 计算梯度矩阵 G 在每个奇异向量方向上的投影
                projections = []
                for i in range(len(S)):
                    u_i = U[:, i]  # 左奇异向量 u_i，shape: (192,)
                    v_i = Vt[i, :]  # 右奇异向量 v_i，shape: (768,)
                    projection = torch.dot(u_i, delta_W @ v_i.T)  # 修正的计算公式 u_i^T G v_i
                    projections.append(projection.item())
                    
                # 将前10个投影值记录到 TensorBoard
                for i, proj_value in enumerate(projections):
                    if i < 5:
                        writer.add_scalar(f'grad_projection/{i}_{name}', proj_value, step)
                
                # Calculate effective rank using the new method
                effective_rank = (S ** 2).sum() / (S[0] ** 2) if S[0] != 0 else 0
                effective_rank = math.ceil(effective_rank)
                
                
                projection_sum = sum(projections[:effective_rank])
                writer.add_scalar(f'grad_projection_sum/{name}', projection_sum, step)

                