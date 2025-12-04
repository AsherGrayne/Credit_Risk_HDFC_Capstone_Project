"""
Batch processing utilities for handling large datasets
"""
import pandas as pd
import numpy as np
from typing import Iterator, Callable, Optional
from src.logging_config import get_logger
from src.monitoring import track_performance, metrics_collector

logger = get_logger(__name__)


class BatchProcessor:
    """Process large datasets in batches"""
    
    def __init__(self, batch_size: int = 1000):
        """
        Initialize batch processor
        
        Args:
            batch_size: Number of records per batch
        """
        self.batch_size = batch_size
    
    def process_in_batches(
        self,
        df: pd.DataFrame,
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        Process dataframe in batches
        
        Args:
            df: Dataframe to process
            process_func: Function to process each batch (takes DataFrame, returns DataFrame)
            progress_callback: Optional callback function(progress_percent, batch_num, total_batches)
        
        Returns:
            Processed dataframe
        """
        total_rows = len(df)
        total_batches = (total_rows + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {total_rows} rows in {total_batches} batches of {self.batch_size}")
        
        results = []
        
        for batch_num in range(total_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_rows)
            
            batch_df = df.iloc[start_idx:end_idx].copy()
            
            # Process batch
            try:
                processed_batch = process_func(batch_df)
                results.append(processed_batch)
                
                # Update progress
                progress = ((batch_num + 1) / total_batches) * 100
                if progress_callback:
                    progress_callback(progress, batch_num + 1, total_batches)
                
                logger.debug(f"Processed batch {batch_num + 1}/{total_batches} ({progress:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num + 1}: {e}")
                raise
        
        # Combine results
        if results:
            final_df = pd.concat(results, ignore_index=True)
            logger.info(f"Successfully processed {len(final_df)} rows")
            return final_df
        else:
            return pd.DataFrame()
    
    @track_performance("batch_processing")
    def process_with_metrics(
        self,
        df: pd.DataFrame,
        process_func: Callable
    ) -> pd.DataFrame:
        """
        Process dataframe with performance metrics
        
        Args:
            df: Dataframe to process
            process_func: Function to process each batch
        
        Returns:
            Processed dataframe
        """
        metrics_collector.increment_counter("batch_processing_started")
        
        try:
            result = self.process_in_batches(df, process_func)
            metrics_collector.increment_counter("batch_processing_completed")
            metrics_collector.record_value("batch_processing_rows", len(result))
            return result
        except Exception as e:
            metrics_collector.increment_counter("batch_processing_errors")
            logger.error(f"Batch processing failed: {e}")
            raise


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
    """
    Generator to yield chunks of dataframe
    
    Args:
        df: Dataframe to chunk
        chunk_size: Size of each chunk
    
    Yields:
        Dataframe chunks
    """
    total_rows = len(df)
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        yield df.iloc[start_idx:end_idx].copy()

