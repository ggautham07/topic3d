# CINES Allocations


==GENOA==: Each scalar (GENOA) node is equipped with **two AMD EPYC 9654 processors**, each providing **96 cores at 2.4 GHz** (192 cores per node). Compared to Zen 3, the Zen 4 architecture now supports AVX-512 though not at full speed (it _emulates_ AVX-512 using AVX2). 768 GiB of DDR5-4800 MHz memory.

==MI250A==: Each accelerated (MI250X) node consists of **1** AMD EPYC 7A53 (Trento) **processor** with **64 cores** at 2.0 GHz and **4** AMD Instinct **MI250X accelerators** (code name gfx90a, Aldebaran, CDNA 2 microarchitecture) as shown in the figure below. This provides 64 cores (with **2 hardware threads per core**) attached to **256 Gio** of DDR4-3200 MHz memory and **8 Graphics Compute Dies** (GCD) per node. The MI250X accelerator is a Multi-Chip Module (MCM) and comes with 2 GCDs. **A GCD can be seen as a GPU**. The user can think of the 8 GCDs as 8 separate GPUs

- **1 AMD EPYC 7A53 CPU** with **64 cores** @ 2.0 GHz
- **4 MI250X cards**, each containing **2 GCDs (Graphics Compute Dies)**
- **Each GPU has 64Gb of memory.**
- **Total: 8 GCDs** that function as separate GPUs
- **256 GiB DDR4** (CPU) + **512 GiB HBM2e** (GPU memory, 64 GiB per GCD)

==MI300A==: Each accelerated (MI300A) node consists of **four AMD Instinct MI300A accelerators** (code name _gfx942_, CDNA 3 microarchitecture). Each MI300A is an **Accelerated Processing Unit (APU)** that integrates both:
- **24 CPU cores** (Zen 4-based);
- **228 CDNA 3 Compute Units (CU)**.
The GPU and CPU shared a unified HBM memory pool. The MI300A is a natural extension to the MCM which is the MI250X. While for the MI250X, producing an IO die that was fast enough to fuse the two GCDs as if they were one big GPU was not feasible, this was rectified on the MI300A/X. Each MI300A is shown as one GPU by `rocm-smi`.

==HPDA==: The 12 preprocessing and postprocessing nodes also called High Performance Data Analytics (HPDA) are based on 2U HPE-Cray Proliant DL385 Gen11 servers with 2 AMD Genoa processors. Each processor has 96 cores @ 2.4 GHz. Each node has 2 NVIDIA L40 graphics cards attached for handling the pre/post processing workloads. Every such node has 2048 Gio of RAM configured in 16 memory channels with 2 x 64 Gio DDR5-4800 MHz memory arrays per channel. For inter-node communications, each node has 1 Slingshot 200 Gb/s network interfaces and a dual-port 10 Gb/s Ethernet card. Theses nodes will be connected to the rest of the machine via the Slingshot fabric and to the CINES facilities via Arista _edge routers_

## A0190316693

Project ID: c1916693
Responsable : M. Marcelo Nollmann martinez 
**Titre du projet : 3D_spatial_genomics**

#### requested
`-150 000 heures cœurs demandées sur Adastra Genoa HPDA NL du CINES`
-100 000 heures cœurs demandées sur Adastra Genoa du CINES
-20 000 heures GPU *demandées* sur Adastra MI250x du CINES
-3 000 heures GPU demandées sur Adastra MI300A du CINES

#### attributed
- 0 heure attribuée sur Adastra Genoa HPDA NL du CINES le mardi 21 octobre 2025
- 250 000 heures cœurs attribuées sur Adastra ==Genoa== du CINES le samedi 01 novembre 2025
- 20 000 heures GPU attribuées sur Adastra ==MI250x== du CINES le samedi 01 novembre 2025
- 3 000 heures GPU attribuées sur Adastra ==MI300A== du CINES le samedi 01 novembre 2025

##### Stats

Based on some initial tests, I need approximately:

1.79 h per ROI (25 cycles) in Genoa.
For a full experiments with 30 ROIs: 193620 s ~ 53h
==For the attributed time for the project: 4600 Hi-M datasets.==

**A dataset with 30 ROIs could then be analyzed in 1.79h using 30 nodes.**

this is without much optimization of node architecture or of pyHiM itself.

## AD010316693R1

Project ID: c10316693R1

Allocation dynamique spatial_genomics

- 35 343 heures cœurs attribuées sur Adastra ==Genoa== du CINES le jeudi 26 juin 2025
- 7 069 heures GPU attribuées sur Adastra MI250x du CINES le jeudi 26 juin 2025
- 53 013 heures cœurs attribuées sur Adastra Genoa HPDA NL du CINES le jeudi 26 juin 2025
- 1 060 heures GPU attribuées sur Adastra MI300A du CINES le jeudi 26 juin 2025