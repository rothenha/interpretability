import React from 'react';
import PropTypes from 'prop-types';

import {Grid, Typography, Tooltip} from '@material-ui/core';

import ReconstructSentence from '../reconstruct/ReconstructSentence';
import GlyphComponent from '../glyph/GlyphComponent';

import * as glyphs from '../../glyphs';

/**
 * Provides a Body Component for the Dreaming Card.
 */
class DreamBody extends React.Component {
  /**
   * Renders the Component.
   *
   * @return {jsx} The component to be rendered.
   */
  render() {
    const iterations = this.props.results.iterations;
    const glyphsParams = glyphs.iterationsToGlyphsParams(iterations);
    return (
      <Grid container direction='column' spacing={2} wrap='nowrap'>
        {iterations.map((iteration, index) =>
          <Grid item key={index}>
            <Grid container direction='row' spacing={1}>
              <Tooltip title="Iteration Number" placement="top">
                <Grid item style={{width: this.props.sentenceParams.itWidth}}>
                  <Typography variant="caption" color="inherit">
                    {iteration.number}
                  </Typography>
                </Grid>
              </Tooltip>
              <Grid item>
                <Grid container direction='row' spacing={0}>
                  {Object.keys(glyphsParams).map((key, idx) =>
                    <Tooltip
                      title={key + ': ' +
                          glyphsParams[key].iterations[
                              index].toFixed(4)}
                      placement="top" key={idx}>
                      <Grid item>
                        <GlyphComponent
                          value={glyphsParams[key].iterations[index]}
                          extremes={glyphsParams[key].extremes}
                          color={glyphsParams[key].color}/>
                      </Grid>
                    </Tooltip>
                  )}
                </Grid>
              </Grid>
              <Grid item>
                <ReconstructSentence sentence={iteration.tokens}
                  target={this.props.sentenceParams.target}
                  original={this.props.sentenceParams.target}
                  colors={this.props.sentenceParams.colors}/>
              </Grid>
            </Grid>
          </Grid>
        )}
      </Grid>
    );
  }
}

DreamBody.propTypes = {
  results: PropTypes.object.isRequired,
  params: PropTypes.object.isRequired,
  sentenceParams: PropTypes.object.isRequired,
};

export default DreamBody;
