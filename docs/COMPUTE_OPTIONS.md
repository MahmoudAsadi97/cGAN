# Compute options for continued Phase 1

Assessment date: 21 October 2026. The current environment has four CPU cores,
32 GiB RAM, and no GPU. It can continue the controlled development experiments.
The next useful purchase decision depends on measured runtime for the chosen
experiment, rather than a promised number of GPU hours for publication.

## Recommendation

Use the existing CPU machine for short diagnostic comparisons. Trial one free
Kaggle or Google Colab GPU session before paying for Azure. This model is a small
fully connected network; GPU acceleration may be limited by small operations,
data transfers, or CPU validation. Measure training and validation separately,
and compare the total elapsed time for the same workload.

| Option | Direct compute charge | Suitability and constraints |
| --- | --- | --- |
| Current CPU environment | No additional Azure or notebook charge; any existing environment charge remains separate | Available now; useful for parallel short probes. |
| Kaggle free notebook accelerator | No compute payment within the account's free allowance | Check the account's current accelerator availability, remaining quota, session limits and verification requirements. Preserve outputs before sessions end. |
| Google Colab free runtime | No compute payment within the free allocation | Accelerator type, availability, usage limits and session duration vary. A requested GPU is not guaranteed. Preserve outputs before disconnection. |
| Azure GPU VM | VM hours plus storage and applicable transfer charges | More control over software and run duration, subject to subscription quota and regional capacity. Spot instances may be interrupted. |

Free notebook quotas are not a commitment of continuously available compute.
Do not plan a full seed study around an unverified fixed weekly allowance. Run
the same declared configurations and retain every attempt, including interrupted
runs, whichever provider supplies the hardware.

## Azure cost calculation

No Azure resources were created, and there is no authenticated Azure connection
configured for this work. A connection would require an authorized subscription
and a secure sign-in or managed identity, followed by a region and available GPU
quota. Credentials should not be pasted into chat or committed to this repository.

A possible small GPU VM to price is `Standard_NC4as_T4_v3`; check its current
regional availability and Linux consumption rate in the Azure calculator or
Retail Prices API. The rate is for the **whole VM per hour**, not a separate
GPU-only charge. A paid run should first have an agreed resource list, hourly
rate, maximum runtime, and shutdown/deallocation procedure.

These are **hypothetical arithmetic examples, not Azure price quotes**:

| Assumed VM rate, USD/hour | 10 VM hours | 50 VM hours |
| ---: | ---: | ---: |
| $0.50 | $5 | $25 |
| $1.00 | $10 | $50 |
| $2.00 | $20 | $100 |

Compute cost equals the current VM hourly rate multiplied by total allocated
VM hours. If five seed jobs run sequentially for two hours each, that is ten VM
hours. Parallel VMs multiply the rate. Add retained disks, applicable data
transfer, taxes and other selected services. A stopped operating system is not
the same as a deallocated VM; confirm the resource's billed state. Budget alerts
alone do not impose a spending cap, and disks can remain billable after compute
is deallocated. Spot rates and interruptions make a fixed-runtime promise
inappropriate without checkpoint recovery.

## GPU pilot and validation status

The repository's default setup deliberately installs `torch==2.8.0+cpu`; it does
enable CUDA training through that setup. The separate
[GPU notebook](GPU_NOTEBOOK.md) reuses a compatible provider CUDA environment,
records its software/hardware versions, and runs a bounded timing pilot.
The development trainer now supports explicit CUDA execution and continuation
of completed training segments into new directories. CPU continuation has been
verified; actual GPU execution and recovery from arbitrary interruptions remain
unverified. Buying GPU time alone will not accelerate the default CPU command.

Before estimating the full study, benchmark a fixed number of updates on the
candidate GPU with the actual batch size, dataset and validation schedule.
Report both update throughput and total wall time. Preserve CPU results and
identify GPU runs by their backend; numerical reproducibility across CPU and
GPU is not assumed to be bitwise identical.

## Official references and verification limit

Read-only requests to the following sources were attempted on 21 October 2026.
The environment's network proxy returned HTTP 403 before retrieval, including
one approved read-only retry. Therefore, **no live Azure rates or current free
GPU quota values have been verified in this document**. The links are the places
to check current account- and region-specific terms, not citations for a fetched
price quote.

- [Azure pricing calculator](https://azure.microsoft.com/en-us/pricing/calculator/)
- [Azure Linux VM pricing](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/)
- [Azure Retail Prices API documentation](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices)
- [Azure Retail Prices API endpoint](https://prices.azure.com/api/retail/prices)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
