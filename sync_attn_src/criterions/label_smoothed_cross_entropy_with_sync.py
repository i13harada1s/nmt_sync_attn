import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion


@register_criterion('label_smoothed_cross_entropy_with_sync')
class LabelSmoothedCrossEntropyCriterionWithSync(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,
        task, 
        sentence_avg,
        label_smoothing,
        sync_lambda,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.sync_lambda = sync_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--sync-lambda', default=1.0, type=float, metavar='D',
                            help='weight for the synchronization loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        # Compute synchronization loss.
        sync_loss = self.compute_sync_loss(net_output)

        if sync_loss is not None:
            logging_output['sync_loss'] = utils.item(sync_loss.data)
            loss += self.sync_lambda * sync_loss

        return loss, sample_size, logging_output

    def compute_sync_loss(self, net_output):
        src_attn_prob = net_output[1][0]
        tgt_attn_prob = net_output[1][1]

        if src_attn_prob.shape == tgt_attn_prob.shape:
            # synchronization loss computation.
            loss = F.mse_loss(src_attn_prob, tgt_attn_prob, reduction="sum")
        else:
            return None

        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(
            sum(log.get('loss', 0) for log in logging_outputs)
        )
        nll_loss_sum = utils.item(
            sum(log.get('nll_loss', 0) for log in logging_outputs)
            )
        sync_loss_sum = utils.item(
            sum(log.get('sync_loss', 0) for log in logging_outputs)
        )
        ntokens = utils.item(
            sum(log.get('ntokens', 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get('sample_size', 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            'loss', loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            'nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            'sync_loss', sync_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            'ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True