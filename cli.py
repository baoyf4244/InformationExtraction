from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser


optimizer_params = {
    'adamw': {
        '--beta1': 0.9,
        '--beta2': 0.98,
        '--eps': 1e-8,
        '--weight_decay': 0.01
    }
}

lr_schedule_params = {
    'polynomial_decay_warmup': {
        '--warmup_steps': 1000,
        '--lr_end': 1e-7,
        '--power': 1.0,
        '--last_epoch': -1
    }
}


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parent_parser: LightningArgumentParser) -> None:
        self.add_optimizer_params(parent_parser)
        self.add_lr_schedule_params(parent_parser)

    @staticmethod
    def add_lr_schedule_params(parent_parser: LightningArgumentParser):
        parser = parent_parser.add_argument_group(name='lr_schedule')
        parser.add_argument('--name', type=str, default='polynomial_decay_warmup')
        temp_args, _ = parent_parser.parse_known_args()
        if temp_args.name in optimizer_params:
            for key, value in optimizer_params[temp_args.name].items():
                parser.add_argument(key, type=type(value), default=value)

    @staticmethod
    def add_optimizer_params(parent_parser: LightningArgumentParser):
        parser = parent_parser.add_argument_group(name='optimizer')
        parser.add_argument('--name', type=str, default='adamw')
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        temp_args, _ = parent_parser.parse_known_args()
        print(temp_args)
        if temp_args.name in optimizer_params:
            for key, value in optimizer_params[temp_args.name].items():
                parser.add_argument(key, type=type(value), default=value)