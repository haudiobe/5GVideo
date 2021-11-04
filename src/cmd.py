import click

@click.group()
@click.option('--dry-run', default=False)
def cli(dry_run):
    click.echo(f"dry-run mode is {'on' if dry_run else 'off'}")

@cli.command()  # @cli, not @click!
def sync():
    click.echo('Syncing')

# filter by anchor key

# filter by scenario + filter by encoder config

# encode

# decode

# metrics

# verify

# compare -a -t

