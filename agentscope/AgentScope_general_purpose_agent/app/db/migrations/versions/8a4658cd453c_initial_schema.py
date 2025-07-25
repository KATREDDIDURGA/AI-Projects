"""initial schema

Revision ID: 8a4658cd453c
Revises: 
Create Date: 2025-07-21 15:28:23.248554

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8a4658cd453c'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('agent_runs',
    sa.Column('run_id', sa.String(), nullable=False),
    sa.Column('agent_type', sa.String(), nullable=True),
    sa.Column('query', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('final_decision', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('run_id')
    )
    op.create_table('agent_metrics',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.String(), nullable=True),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('value', sa.Float(), nullable=True),
    sa.Column('unit', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['agent_runs.run_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_metrics_id'), 'agent_metrics', ['id'], unique=False)
    op.create_table('agent_steps',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('run_id', sa.String(), nullable=True),
    sa.Column('step_number', sa.Integer(), nullable=True),
    sa.Column('thought', sa.Text(), nullable=True),
    sa.Column('observation', sa.Text(), nullable=True),
    sa.Column('action', sa.Text(), nullable=True),
    sa.Column('result', sa.Text(), nullable=True),
    sa.Column('timestamp', sa.DateTime(), nullable=True),
    sa.Column('severity', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['agent_runs.run_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_steps_id'), 'agent_steps', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_agent_steps_id'), table_name='agent_steps')
    op.drop_table('agent_steps')
    op.drop_index(op.f('ix_agent_metrics_id'), table_name='agent_metrics')
    op.drop_table('agent_metrics')
    op.drop_table('agent_runs')
    # ### end Alembic commands ###
